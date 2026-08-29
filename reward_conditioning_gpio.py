#!/usr/bin/env python3
"""Independent GPIO process for reward delivery and lick capture.

Why a separate process is required
----------------------------------
The current ``rpg.Screen.display_raw()`` call enters a C extension and blocks
for the complete raw sequence without releasing Python's GIL.  A gpiozero
callback thread or ``LED.blink(background=True)`` in the visual-stimulus
process can therefore be delayed while an image is displayed.  This module
places all lick callbacks and reward pulse timing in a spawned child process.

Reward delivery is open-loop.  A reward command is executed solely because the
precomputed trial plan marks the trial as rewarded.  Lick state is never read
when deciding whether to open the valve.
"""

from __future__ import print_function

import multiprocessing
import signal
import threading
import time
from datetime import datetime, timezone


def _unix_ns_to_iso(unix_ns):
    seconds, nanoseconds = divmod(int(unix_ns), 1_000_000_000)
    dt = datetime.fromtimestamp(seconds, timezone.utc)
    return "%s.%09d+00:00" % (dt.strftime("%Y-%m-%dT%H:%M:%S"), nanoseconds)


def _unix_ns_to_seconds_string(unix_ns):
    seconds, nanoseconds = divmod(int(unix_ns), 1_000_000_000)
    return "%d.%09d" % (seconds, nanoseconds)


def _timestamp_fields():
    unix_ns = time.time_ns()
    monotonic_ns = time.monotonic_ns()
    return {
        "utc_iso": _unix_ns_to_iso(unix_ns),
        "unix_time_utc_sec": _unix_ns_to_seconds_string(unix_ns),
        "unix_time_ns": unix_ns,
        "monotonic_ns": monotonic_ns,
    }


def _event(event_type, context=None, **extra):
    row = {
        "message_type": "event",
        "event_type": event_type,
    }
    row.update(_timestamp_fields())
    if context:
        row.update(dict(context))
    row.update(extra)
    return row


class _MockLED(object):
    """Small simulation-only gpiozero LED replacement."""

    def __init__(self, pin):
        self.pin = pin
        self.value = 0

    def on(self):
        self.value = 1

    def off(self):
        self.value = 0

    def close(self):
        self.off()


def gpio_worker_main(connection, config):
    """Own the reward valve and lick detector until a shutdown command arrives."""
    # The parent owns terminal Ctrl-C and is responsible for orderly cleanup.
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass
    send_lock = threading.Lock()
    context_lock = threading.Lock()
    current_context = {
        "phase": "startup",
        "trial_index": "",
        "trial_number": "",
        "block_number": "",
        "image_role": "",
        "image_filename": "",
        "reward_scheduled": "",
        "suction_scheduled": "",
    }
    reward_led = None
    suction_led = None
    lick_button = None

    def safe_send(message):
        with send_lock:
            connection.send(message)

    def context_snapshot():
        with context_lock:
            return dict(current_context)

    def log_lick_onset():
        # The existing behavior-box wiring calls Button.when_released a lick
        # "entry".  Preserve that convention here.
        safe_send(
            _event(
                "lick_onset",
                context_snapshot(),
                lick_pin_bcm=config["lick_pin_bcm"],
                lick_edge="button_when_released",
            )
        )

    def log_lick_offset():
        safe_send(
            _event(
                "lick_offset",
                context_snapshot(),
                lick_pin_bcm=config["lick_pin_bcm"],
                lick_edge="button_when_pressed",
            )
        )

    try:
        simulate = bool(config.get("simulate_gpio", False))
        if simulate:
            reward_led = _MockLED(config["reward_pin_bcm"])
            suction_led = _MockLED(config["suction_pin_bcm"])
        else:
            from gpiozero import Button, LED

            reward_led = LED(config["reward_pin_bcm"])
            reward_led.off()
            suction_led = LED(config["suction_pin_bcm"])
            suction_led.off()
            lick_button = Button(
                config["lick_pin_bcm"],
                pull_up=None,
                active_state=True,
                bounce_time=config.get("lick_bounce_time_sec", None),
            )
            lick_button.when_released = log_lick_onset
            lick_button.when_pressed = log_lick_offset

        safe_send(
            {
                "message_type": "ready",
                "worker_pid": multiprocessing.current_process().pid,
                "simulate_gpio": simulate,
                "reward_pin_bcm": config["reward_pin_bcm"],
                "lick_pin_bcm": config["lick_pin_bcm"],
                "suction_pin_bcm": config["suction_pin_bcm"],
            }
        )
        safe_send(_event("gpio_worker_ready", context_snapshot()))

        def deliver_pulse_train(
            reward_context,
            command_id,
            command_event_type,
            complete_event_type,
            trigger_source,
        ):
            """Run valve timing first; IPC logging is deadline-compensated."""
            pulse_on_sec = float(config["reward_pulse_on_sec"])
            pulse_off_sec = float(config["reward_pulse_off_sec"])
            num_pulses = int(config["reward_num_pulses"])
            command_event = _event(
                command_event_type,
                reward_context,
                command_id=command_id,
                reward_pin_bcm=config["reward_pin_bcm"],
                reward_pulse_on_sec=pulse_on_sec,
                reward_pulse_off_sec=pulse_off_sec,
                reward_num_pulses=num_pulses,
                reward_trigger_source=trigger_source,
            )

            for pulse_index in range(1, num_pulses + 1):
                on_deadline = time.monotonic() + pulse_on_sec
                reward_led.on()
                valve_on_event = _event(
                    "reward_valve_on",
                    reward_context,
                    command_id=command_id,
                    reward_pin_bcm=config["reward_pin_bcm"],
                    reward_pulse_index=pulse_index,
                    reward_num_pulses=num_pulses,
                )
                if pulse_index == 1:
                    # The command-received timestamp was captured before the
                    # valve opened, but no pipe write was allowed to delay the
                    # first valve transition.
                    safe_send(command_event)
                safe_send(valve_on_event)
                remaining_on = on_deadline - time.monotonic()
                if remaining_on > 0:
                    time.sleep(remaining_on)

                reward_led.off()
                valve_off_event = _event(
                    "reward_valve_off",
                    reward_context,
                    command_id=command_id,
                    reward_pin_bcm=config["reward_pin_bcm"],
                    reward_pulse_index=pulse_index,
                    reward_num_pulses=num_pulses,
                )
                off_deadline = time.monotonic() + pulse_off_sec
                safe_send(valve_off_event)

                if pulse_index < num_pulses and pulse_off_sec > 0:
                    remaining_off = off_deadline - time.monotonic()
                    if remaining_off > 0:
                        time.sleep(remaining_off)

            safe_send(
                _event(
                    complete_event_type,
                    reward_context,
                    command_id=command_id,
                    reward_pin_bcm=config["reward_pin_bcm"],
                    reward_num_pulses=num_pulses,
                )
            )

        def deliver_suction(suction_context, command_id, trigger_source):
            duration = float(config["suction_duration_sec"])
            received = _event("suction_command_received", suction_context,
                               command_id=command_id, suction_pin_bcm=config["suction_pin_bcm"],
                               suction_duration_sec=duration, suction_trigger_source=trigger_source)
            suction_off_deadline = time.monotonic() + duration
            suction_led.on()
            safe_send(received)
            safe_send(_event("suction_on", suction_context, command_id=command_id,
                             suction_pin_bcm=config["suction_pin_bcm"]))
            remaining = suction_off_deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            suction_led.off()
            safe_send(_event("suction_off", suction_context, command_id=command_id,
                             suction_pin_bcm=config["suction_pin_bcm"]))
            safe_send(_event("suction_complete", suction_context, command_id=command_id,
                             suction_pin_bcm=config["suction_pin_bcm"], suction_duration_sec=duration))

        running = True
        simulated_lick_deadlines = []
        while running:
            # In simulation, poll until either a parent command arrives or a
            # pending synthetic lick deadline is due.  Real GPIO remains on
            # the blocking command path; its callbacks are independent.
            if simulated_lick_deadlines:
                timeout = max(0.0, min(simulated_lick_deadlines) - time.monotonic())
                if not connection.poll(timeout):
                    now = time.monotonic()
                    due = [
                        deadline for deadline in simulated_lick_deadlines
                        if deadline <= now
                    ]
                    simulated_lick_deadlines = [
                        deadline for deadline in simulated_lick_deadlines
                        if deadline > now
                    ]
                    for _deadline in due:
                        safe_send(
                            _event(
                                "lick_onset",
                                context_snapshot(),
                                lick_pin_bcm=config["lick_pin_bcm"],
                                lick_edge="simulated_behavior_test",
                                notes="synthetic_anticipatory_behavior_validation",
                            )
                        )
                    continue
            else:
                connection.poll(None)
            command = connection.recv()
            command_type = command.get("command")
            command_id = command.get("command_id", "")

            if command_type == "set_context":
                with context_lock:
                    current_context.clear()
                    current_context.update(command.get("context", {}))
                safe_send(
                    {
                        "message_type": "ack",
                        "command": "set_context",
                        "command_id": command_id,
                    }
                )

            elif command_type == "reward":
                reward_context = dict(command.get("context", context_snapshot()))
                deliver_pulse_train(
                    reward_context,
                    command_id,
                    "reward_command_received",
                    "reward_complete",
                    "precomputed_open_loop_schedule",
                )

            elif command_type == "manual_reward":
                # Manual priming/calibration uses the same pulse engine but is
                # explicitly labeled as a non-trial event.
                manual_context = context_snapshot()
                manual_context.update(
                    {
                        "phase": "manual_reward",
                        "trial_index": "",
                        "trial_number": "",
                        "image_role": "",
                        "image_filename": "",
                        "reward_scheduled": True,
                    }
                )
                deliver_pulse_train(
                    manual_context,
                    command_id,
                    "manual_reward_command_received",
                    "manual_reward_complete",
                    "operator_manual_test",
                )

            elif command_type == "trigger_suction":
                suction_context = dict(command.get("context", context_snapshot()))
                deliver_suction(suction_context, command_id, "precomputed_reward_associated_schedule")

            elif command_type == "manual_suction":
                manual_context = context_snapshot()
                manual_context.update({"phase": "manual_suction", "trial_index": "",
                                       "trial_number": "", "block_number": "",
                                       "image_role": "", "image_filename": "",
                                       "reward_scheduled": False, "suction_scheduled": False})
                deliver_suction(manual_context, command_id, "operator_manual_test")

            elif command_type == "simulate_lick":
                if not simulate:
                    raise RuntimeError("Synthetic licks require simulate_gpio.")
                safe_send(
                    _event(
                        "lick_onset",
                        context_snapshot(),
                        lick_pin_bcm=config["lick_pin_bcm"],
                        lick_edge="simulated_test",
                        notes="synthetic_water_accounting_validation",
                    )
                )

            elif command_type == "schedule_simulated_lick":
                if not simulate:
                    raise RuntimeError("Synthetic licks require simulate_gpio.")
                delay_sec = float(command.get("delay_sec", 0.0))
                if delay_sec < 0:
                    raise ValueError("Synthetic lick delay must be nonnegative.")
                simulated_lick_deadlines.append(time.monotonic() + delay_sec)
                safe_send(
                    {
                        "message_type": "ack",
                        "command": "schedule_simulated_lick",
                        "command_id": command_id,
                    }
                )

            elif command_type == "shutdown":
                running = False
                safe_send(
                    {
                        "message_type": "ack",
                        "command": "shutdown",
                        "command_id": command_id,
                    }
                )

            else:
                safe_send(
                    _event(
                        "gpio_unknown_command",
                        context_snapshot(),
                        command_id=command_id,
                        command_repr=repr(command),
                    )
                )

    except EOFError:
        pass
    except BaseException as exc:
        try:
            safe_send(
                {
                    "message_type": "fatal",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        except Exception:
            pass
        raise
    finally:
        if reward_led is not None:
            try:
                reward_led.off()
            except Exception:
                pass
            try:
                reward_led.close()
            except Exception:
                pass
        if suction_led is not None:
            try:
                suction_led.off()
                suction_led.close()
            except Exception:
                pass
        if lick_button is not None:
            try:
                lick_button.close()
            except Exception:
                pass
        try:
            safe_send(_event("gpio_worker_stopped", context_snapshot()))
        except Exception:
            pass
        try:
            connection.close()
        except Exception:
            pass


class BehaviorGPIOClient(object):
    """Parent-process interface to the GPIO worker."""

    def __init__(self, config):
        self.config = dict(config)
        self._ctx = multiprocessing.get_context("spawn")
        self._connection = None
        self._process = None
        self._buffered_events = []
        self._command_counter = 0

    def _next_command_id(self, prefix):
        self._command_counter += 1
        return "%s_%06d" % (prefix, self._command_counter)

    def start(self, timeout_sec=10.0):
        if self._process is not None:
            raise RuntimeError("GPIO worker has already been started.")
        parent_connection, child_connection = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(
            target=gpio_worker_main,
            args=(child_connection, self.config),
            name="reward-conditioning-gpio",
        )
        process.daemon = True
        process.start()
        child_connection.close()
        self._connection = parent_connection
        self._process = process

        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            if self._connection.poll(0.05):
                message = self._connection.recv()
                if message.get("message_type") == "ready":
                    return message
                self._store_or_raise(message)
            if not self._process.is_alive():
                raise RuntimeError("GPIO worker exited before reporting ready.")
        raise RuntimeError("Timed out waiting for GPIO worker readiness.")

    def _store_or_raise(self, message):
        message_type = message.get("message_type")
        if message_type == "event":
            self._buffered_events.append(message)
            return
        if message_type == "fatal":
            raise RuntimeError(
                "GPIO worker failed (%s): %s"
                % (message.get("error_type", "error"), message.get("error", ""))
            )
        # Unexpected acknowledgements are retained so diagnostics are not lost.
        self._buffered_events.append(
            {
                "message_type": "event",
                "event_type": "gpio_unexpected_message",
                **_timestamp_fields(),
                "message_repr": repr(message),
            }
        )

    def _wait_for_ack(self, command, command_id, timeout_sec=5.0):
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            if self._connection.poll(0.05):
                message = self._connection.recv()
                if (
                    message.get("message_type") == "ack"
                    and message.get("command") == command
                    and message.get("command_id") == command_id
                ):
                    return message
                self._store_or_raise(message)
            if self._process is not None and not self._process.is_alive():
                raise RuntimeError("GPIO worker exited while waiting for %s acknowledgement." % command)
        raise RuntimeError("Timed out waiting for GPIO %s acknowledgement." % command)

    def set_context(self, context, timeout_sec=5.0):
        command_id = self._next_command_id("context")
        self._connection.send(
            {
                "command": "set_context",
                "command_id": command_id,
                "context": dict(context),
            }
        )
        return self._wait_for_ack("set_context", command_id, timeout_sec=timeout_sec)

    def trigger_reward(self, trial_context):
        """Send an unconditional reward command and return its command ID."""
        command_id = self._next_command_id("reward")
        self._connection.send(
            {
                "command": "reward",
                "command_id": command_id,
                "context": dict(trial_context),
            }
        )
        return command_id

    def trigger_suction(self, trial_context):
        command_id = self._next_command_id("suction")
        self._connection.send({"command": "trigger_suction", "command_id": command_id,
                               "context": dict(trial_context)})
        return command_id

    def manual_suction(self):
        command_id = self._next_command_id("manual_suction")
        self._connection.send({"command": "manual_suction", "command_id": command_id})
        return command_id

    def manual_reward(self):
        command_id = self._next_command_id("manual_reward")
        self._connection.send(
            {
                "command": "manual_reward",
                "command_id": command_id,
            }
        )
        return command_id

    def simulate_lick(self):
        """Inject one test lick only when the GPIO worker is simulated."""
        if not bool(self.config.get("simulate_gpio", False)):
            raise RuntimeError(
                "Synthetic licks require --simulate-gpio; refusing real GPIO mode."
            )
        command_id = self._next_command_id("simulate_lick")
        self._connection.send({
            "command": "simulate_lick",
            "command_id": command_id,
        })
        return command_id

    def schedule_simulated_lick(self, delay_sec):
        """Schedule one asynchronous test lick in the simulated GPIO child."""
        if not bool(self.config.get("simulate_gpio", False)):
            raise RuntimeError(
                "Synthetic licks require --simulate-gpio; refusing real GPIO mode."
            )
        delay_sec = float(delay_sec)
        if delay_sec < 0:
            raise ValueError("Synthetic lick delay must be nonnegative.")
        command_id = self._next_command_id("schedule_simulate_lick")
        self._connection.send({
            "command": "schedule_simulated_lick",
            "command_id": command_id,
            "delay_sec": delay_sec,
        })
        self._wait_for_ack("schedule_simulated_lick", command_id)
        return command_id

    def drain_events(self):
        events = list(self._buffered_events)
        self._buffered_events = []
        if self._connection is None:
            return events
        while self._connection.poll(0):
            message = self._connection.recv()
            if message.get("message_type") == "event":
                events.append(message)
            else:
                self._store_or_raise(message)
        if self._buffered_events:
            events.extend(self._buffered_events)
            self._buffered_events = []
        return events

    def is_alive(self):
        return self._process is not None and self._process.is_alive()

    def shutdown(self, timeout_sec=5.0):
        if self._process is None:
            return []
        events = self.drain_events()
        if self._process.is_alive():
            command_id = self._next_command_id("shutdown")
            try:
                self._connection.send(
                    {
                        "command": "shutdown",
                        "command_id": command_id,
                    }
                )
                self._wait_for_ack("shutdown", command_id, timeout_sec=timeout_sec)
            except Exception:
                # Continue to join/terminate so the valve cannot be left under an
                # orphaned process after an exception.
                pass
        self._process.join(timeout=float(timeout_sec))
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        try:
            events.extend(self.drain_events())
        except Exception:
            pass
        try:
            self._connection.close()
        except Exception:
            pass
        self._connection = None
        self._process = None
        return events


if __name__ == "__main__":
    # Run as a real file (not ``python -c``) so multiprocessing spawn can import it.
    example_config = {
        "simulate_gpio": True,
        "reward_pin_bcm": 19,
        "suction_pin_bcm": 25,
        "lick_pin_bcm": 26,
        "suction_duration_sec": 0.01,
        "lick_bounce_time_sec": 0.003,
        "reward_pulse_on_sec": 0.002,
        "reward_pulse_off_sec": 0.001,
        "reward_num_pulses": 3,
    }
    client = BehaviorGPIOClient(example_config)
    print("READY", client.start())
    context = {
        "phase": "trial",
        "trial_index": 0,
        "trial_number": 1,
        "block_number": 1,
        "image_role": "rewarded_high_1",
        "image_filename": "example.png",
        "reward_scheduled": True,
        "suction_scheduled": True,
    }
    client.set_context(context)
    print("COMMAND", client.trigger_reward(context))
    time.sleep(0.05)
    for item in client.drain_events():
        print(item)
    for item in client.shutdown():
        print(item)
