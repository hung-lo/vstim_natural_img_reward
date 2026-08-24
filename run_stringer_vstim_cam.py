#!/usr/bin/env python3
"""
run_stringer_vstim_cam.py

Camera-enabled wrapper around run_stringer_vstim.py.
This keeps the original stimulus runner untouched while adding explicit
remote camera start/stop control for the second Pi.
"""

import sys

if __name__ == "__main__":
    print(
        "This repository's supported experiment entrypoint is:\n\n"
        "  python3 run_stringer_reward_conditioning.py\n\n"
        "For the natural-image-only protocol use:\n"
        "  https://github.com/hung-lo/vstim_natural",
        file=sys.stderr,
    )
    raise SystemExit(2)

import json
import math
import select
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import run_stringer_vstim as base

PROJECT_ROOT = base.PROJECT_ROOT
OUTPUT_ROOT = base.OUTPUT_ROOT
CAMERA_CONTROL_SCRIPT = PROJECT_ROOT / "remote_camera_control.py"
DEFAULT_PRESTIM_BASELINE_MINUTES = 3.0
BASELINE_INPUT_POLL_SEC = 0.25
BASELINE_STATUS_INTERVAL_SEC = 10.0
CAMERA_CONTROL_START_HARD_TIMEOUT_SEC = 90.0
# Emergency wrapper timeout only; the controller owns the normal launch,
# readiness, diagnostics, and verified-cleanup timeouts. Keep this longer
# than their combined worst-case path so the wrapper does not interrupt
# camera cleanup.
CAMERA_CONTROL_FETCH_HARD_TIMEOUT_SEC = 600.0
CAMERA_CONTROL_RESULT_PREFIX = "CAMERA_CONTROL_RESULT_JSON="


def subprocess_output_to_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def tail_text(text, max_chars=4000):
    text = text or ""
    if len(text) <= max_chars:
        return text
    return "...[truncated]...\n" + text[-max_chars:]


def _log_completed_process(proc, label):
    stdout, stderr = proc.communicate()
    if stdout:
        print(stdout, end="")
    if stderr:
        print(stderr, end="", file=sys.stderr)
    if proc.returncode not in (0, None):
        print("%s exited with code %s" % (label, proc.returncode), file=sys.stderr)


def run_camera_control(args, background=False, timeout=None):
    cmd = [sys.executable, str(CAMERA_CONTROL_SCRIPT)] + list(args)
    print("+ " + " ".join(shlex.quote(x) for x in cmd))
    if background:
        proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        thread = threading.Thread(target=_log_completed_process, args=(proc, "camera control"), daemon=True)
        thread.start()
        return proc

    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stdout_text = subprocess_output_to_text(exc.stdout)
        stderr_text = subprocess_output_to_text(exc.stderr)
        if stdout_text:
            print(stdout_text, end="" if stdout_text.endswith("\n") else "\n")
        if stderr_text:
            print(stderr_text, end="" if stderr_text.endswith("\n") else "\n", file=sys.stderr)
        timeout_desc = "unknown" if timeout is None else "%.1f" % timeout
        detail_lines = []
        if stdout_text:
            detail_lines.append("stdout tail:\n%s" % tail_text(stdout_text))
        if stderr_text:
            detail_lines.append("stderr tail:\n%s" % tail_text(stderr_text))
        message = "Camera controller exceeded the %s-second emergency timeout." % timeout_desc
        if detail_lines:
            message += "\n" + "\n".join(detail_lines)
        raise RuntimeError(message) from exc
    except subprocess.CalledProcessError as exc:
        stdout_text = subprocess_output_to_text(exc.stdout)
        stderr_text = subprocess_output_to_text(exc.stderr)
        if stdout_text:
            print(stdout_text, end="" if stdout_text.endswith("\n") else "\n")
        if stderr_text:
            print(stderr_text, end="" if stderr_text.endswith("\n") else "\n", file=sys.stderr)
        raise

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result


def parse_camera_control_result(result):
    for line in reversed((result.stdout or "").splitlines()):
        if line.startswith(CAMERA_CONTROL_RESULT_PREFIX):
            return json.loads(line[len(CAMERA_CONTROL_RESULT_PREFIX):])
    raise RuntimeError("Camera controller did not return a structured result.")


def require_camera_state(result, field, operation):
    state = parse_camera_control_result(result)
    if not state.get(field):
        raise RuntimeError("Camera %s was not confirmed by the controller." % operation)
    return state


def start_camera_recording(mouse_id, session_id):
    print("Starting remote camera recording...")
    result = run_camera_control(
        ["start", "--mouse-id", mouse_id, "--session-id", session_id, "--json"],
        background=False,
        timeout=CAMERA_CONTROL_START_HARD_TIMEOUT_SEC,
    )
    state = require_camera_state(result, "camera_output_growing_confirmed", "video output growth")
    if not state.get("camera_pid_confirmed"):
        raise RuntimeError("Camera process was not confirmed by the controller.")
    return state


def stop_camera_recording():
    print("Stopping remote camera recording...")
    result = run_camera_control(["stop", "--json"])
    return require_camera_state(result, "camera_stop_confirmed", "stop")


def fetch_camera_recording():
    print("Fetching camera files to box 151...")
    result = run_camera_control(["fetch", "--json"], timeout=CAMERA_CONTROL_FETCH_HARD_TIMEOUT_SEC)
    return parse_camera_control_result(result)


def convert_camera_recording():
    print("Converting fetched camera files to MP4...")
    result = run_camera_control(["convert", "--json"], timeout=CAMERA_CONTROL_FETCH_HARD_TIMEOUT_SEC)
    return parse_camera_control_result(result)


def start_camera_recording_with_recovery(mouse_id, session_id, event_log_path):
    try:
        state = start_camera_recording(mouse_id, session_id)
        return {
            "confirmed_running": True,
            "controller_state": state,
            "cleanup_attempted": False,
            "cleanup_confirmed": False,
            "camera_state_unknown_acknowledged": False,
            "error": "",
        }
    except (Exception, KeyboardInterrupt) as exc:
        print("Camera start failed: %s" % exc, file=sys.stderr)
        result = {
            "confirmed_running": False,
            "controller_state": {},
            "cleanup_attempted": True,
            "cleanup_confirmed": False,
            "camera_state_unknown_acknowledged": False,
            "error": str(exc),
        }
        base.append_csv_row(
            event_log_path,
            {
                "event_type": "camera_start_failed",
                "notes": "gray_active=true; error=%s" % exc,
            },
            base.EVENT_FIELDS,
        )

        while not result["cleanup_confirmed"] and not result["camera_state_unknown_acknowledged"]:
            cleanup_error = None
            try:
                stop_state = stop_camera_recording()
                result["cleanup_confirmed"] = True
                result["controller_state"] = stop_state
                base.append_csv_row(
                    event_log_path,
                    {
                        "event_type": "camera_cleanup_after_start_failure_confirmed",
                        "notes": "gray_active=true; camera_stop_confirmed=true",
                    },
                    base.EVENT_FIELDS,
                )
                break
            except KeyboardInterrupt:
                cleanup_error = KeyboardInterrupt("interrupted while verifying camera cleanup")
            except Exception as exc_cleanup:
                cleanup_error = exc_cleanup

            print("Could not confirm camera cleanup after start failure: %s" % cleanup_error, file=sys.stderr)
            base.append_csv_row(
                event_log_path,
                {
                    "event_type": "camera_cleanup_after_start_failure_unverified",
                    "notes": "gray_active=true; error=%s" % cleanup_error,
                },
                base.EVENT_FIELDS,
            )

            try:
                retry_cleanup = base.prompt_yes_no(
                    "Retry camera cleanup while keeping the screen gray",
                    default_yes=True,
                )
            except KeyboardInterrupt:
                print("Keyboard interrupt while choosing whether to retry camera cleanup.")
                continue
            if retry_cleanup:
                continue

            try:
                acknowledge_unknown = base.prompt_yes_no(
                    "Acknowledge that the camera may still be running and abort the experiment",
                    default_yes=False,
                )
            except KeyboardInterrupt:
                print("Keyboard interrupt while acknowledging the uncertain camera state.")
                continue
            if acknowledge_unknown:
                result["camera_state_unknown_acknowledged"] = True
                base.append_csv_row(
                    event_log_path,
                    {
                        "event_type": "camera_state_unknown_acknowledged",
                        "notes": "gray_active=true; experiment_aborted=true",
                    },
                    base.EVENT_FIELDS,
                )
                break

            print(
                "The camera must be confirmed stopped or explicitly acknowledged as uncertain. "
                "The screen will remain gray."
            )

        return result




def prompt_float_or_default(prompt, default_value, minimum=0.0):
    raw = base.prompt_text("%s [%g]: " % (prompt, default_value)).strip()
    if not raw:
        return default_value
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("%s must be a finite number" % prompt)
    if value < minimum:
        raise ValueError("%s must be at least %g" % (prompt, minimum))
    return value


def is_early_start_command(line):
    return line.strip().lower() in {"y", "yes"}


def watch_for_early_start(force_start_event, stop_event):
    try:
        stdin = sys.stdin
        if not hasattr(stdin, "isatty") or not stdin.isatty():
            return
    except Exception:
        return

    while not stop_event.is_set() and not force_start_event.is_set():
        try:
            readable, _, _ = select.select([sys.stdin], [], [], BASELINE_INPUT_POLL_SEC)
        except (OSError, ValueError, AttributeError):
            return

        if stop_event.is_set() or force_start_event.is_set():
            return
        if not readable:
            continue

        try:
            line = sys.stdin.readline()
        except Exception:
            return

        if line == "":
            return

        if is_early_start_command(line):
            print(
                "Early start requested. Stimuli will begin as soon as raw preparation and the initial gray period are complete."
            )
            force_start_event.set()
            return

        stripped = line.strip()
        if stripped:
            print(
                "Ignoring input %r; baseline continues. Type y and Enter to request early stimulus start." % stripped
            )


def start_prestimulus_early_start_monitor():
    force_start_event = threading.Event()
    stop_event = threading.Event()
    thread = None
    interactive = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
    if interactive:
        thread = threading.Thread(target=watch_for_early_start, args=(force_start_event, stop_event), daemon=True)
        thread.start()
    else:
        print("Interactive early-start override is unavailable because stdin is not a TTY.")
    return force_start_event, stop_event, thread, interactive


def stop_prestimulus_early_start_monitor(stop_event, thread):
    if stop_event is not None:
        stop_event.set()
    if thread is not None:
        thread.join(timeout=1.0)


def wait_for_prestimulus_gate(
    baseline_start_monotonic,
    requested_baseline_sec,
    gray_start_monotonic,
    minimum_gray_sec,
    force_start_event,
):
    entry_monotonic = time.monotonic()
    remaining_camera_baseline_sec_at_gate_entry = max(
        0.0,
        requested_baseline_sec - (entry_monotonic - baseline_start_monotonic),
    )
    remaining_minimum_gray_sec_at_gate_entry = max(
        0.0,
        minimum_gray_sec - (entry_monotonic - gray_start_monotonic),
    )
    result = {
        "requested_sec": requested_baseline_sec,
        "minimum_gray_sec": minimum_gray_sec,
        "remaining_camera_baseline_sec_at_gate_entry": remaining_camera_baseline_sec_at_gate_entry,
        "remaining_minimum_gray_sec_at_gate_entry": remaining_minimum_gray_sec_at_gate_entry,
        "camera_baseline_elapsed_sec": 0.0,
        "gray_elapsed_sec": 0.0,
        "forced": False,
        "end_reason": "",
        "waited_for_minimum_gray_after_override": False,
    }

    baseline_completed_during_preparation = remaining_camera_baseline_sec_at_gate_entry <= 0.0
    if baseline_completed_during_preparation:
        result["end_reason"] = "timer_satisfied_during_preparation"
    elif force_start_event.is_set():
        result["forced"] = True
        result["end_reason"] = "user_override"
    else:
        result["end_reason"] = "timer_elapsed"

    while True:
        now_monotonic = time.monotonic()
        camera_baseline_elapsed_sec = now_monotonic - baseline_start_monotonic
        gray_elapsed_sec = now_monotonic - gray_start_monotonic
        baseline_ready = force_start_event.is_set() or camera_baseline_elapsed_sec >= requested_baseline_sec
        gray_ready = gray_elapsed_sec >= minimum_gray_sec

        if baseline_ready and gray_ready:
            break

        if force_start_event.is_set():
            result["forced"] = True
            if not gray_ready:
                result["waited_for_minimum_gray_after_override"] = True
                wait_sec = min(BASELINE_STATUS_INTERVAL_SEC, max(0.0, minimum_gray_sec - gray_elapsed_sec))
                if wait_sec > 0.0:
                    time.sleep(wait_sec)
                    continue
            break

        wait_candidates = []
        if not baseline_ready:
            wait_candidates.append(max(0.0, requested_baseline_sec - camera_baseline_elapsed_sec))
        if not gray_ready:
            wait_candidates.append(max(0.0, minimum_gray_sec - gray_elapsed_sec))

        wait_sec = min([BASELINE_STATUS_INTERVAL_SEC] + wait_candidates) if wait_candidates else BASELINE_STATUS_INTERVAL_SEC
        if wait_sec <= 0.0:
            continue

        if force_start_event.wait(timeout=wait_sec):
            if camera_baseline_elapsed_sec < requested_baseline_sec and not baseline_completed_during_preparation:
                result["forced"] = True
                result["end_reason"] = "user_override"
            continue

        now_monotonic = time.monotonic()
        camera_baseline_elapsed_sec = now_monotonic - baseline_start_monotonic
        gray_elapsed_sec = now_monotonic - gray_start_monotonic
        if baseline_completed_during_preparation:
            result["end_reason"] = "timer_satisfied_during_preparation"
        elif camera_baseline_elapsed_sec >= requested_baseline_sec and not force_start_event.is_set():
            result["end_reason"] = "timer_elapsed"
        elif camera_baseline_elapsed_sec < requested_baseline_sec and result["end_reason"] != "user_override":
            result["end_reason"] = "timer_elapsed"

        if not gray_ready and camera_baseline_elapsed_sec >= requested_baseline_sec:
            print("Pre-stimulus gray remaining: %s ..." % base.format_seconds(max(0.0, minimum_gray_sec - gray_elapsed_sec)))
        elif not baseline_ready:
            print("Pre-stimulus baseline remaining: %s ..." % base.format_seconds(max(0.0, requested_baseline_sec - camera_baseline_elapsed_sec)))

    result["camera_baseline_elapsed_sec"] = time.monotonic() - baseline_start_monotonic
    result["gray_elapsed_sec"] = time.monotonic() - gray_start_monotonic
    if not result["end_reason"]:
        result["end_reason"] = "timer_elapsed"
    return result


def prestimulus_result_message(baseline_result):
    reason = baseline_result["end_reason"]
    if reason == "user_override":
        return "Pre-stimulus baseline ended early by user request. Starting visual stimulus playback..."
    if reason == "timer_satisfied_during_preparation":
        return "Requested pre-stimulus baseline was satisfied during preparation. Starting visual stimulus playback..."
    return "Pre-stimulus baseline complete. Starting visual stimulus playback..."


def run_poststim_black_baseline(screen, iti_raw, event_log_path, stimulus_playback_completed=True):
    final_gray_repeats = max(1, int(math.ceil(base.POSTSTIM_GRAY_PLANNED_SEC / base.ITI_DURATION_SEC)))
    poststim_gray_start_utc = base.utc_iso_now()
    base.append_csv_row(
        event_log_path,
        {
            "event_type": "poststim_gray_start",
            "notes": "planned_duration_sec=%.3f; final_trial_iti_included=false; gray_repeats=%d"
            % (base.POSTSTIM_GRAY_PLANNED_SEC, final_gray_repeats),
        },
        base.EVENT_FIELDS,
    )

    poststim_gray_start_monotonic = time.monotonic()
    for _ in range(final_gray_repeats):
        screen.display_raw(iti_raw)

    poststim_gray_actual_sec = time.monotonic() - poststim_gray_start_monotonic
    base.append_csv_row(
        event_log_path,
        {
            "event_type": "poststim_gray_end",
            "notes": "planned_duration_sec=%.3f; actual_sec=%.6f"
            % (base.POSTSTIM_GRAY_PLANNED_SEC, poststim_gray_actual_sec),
        },
        base.EVENT_FIELDS,
    )

    poststim_black_start_monotonic = time.monotonic()
    poststim_black_request_utc = base.utc_iso_now()
    screen.display_greyscale(base.POSTSTIM_BLACK_LEVEL)
    poststim_black_on_utc = base.utc_iso_now()
    base.append_csv_row(
        event_log_path,
        {
            "event_type": "poststim_black_on",
            "notes": "black_level=%d; screen_open=true; request_utc=%s"
            % (base.POSTSTIM_BLACK_LEVEL, poststim_black_request_utc),
        },
        base.EVENT_FIELDS,
    )

    camera_stop_confirmed = False
    camera_left_running_by_user = False
    camera_fetch_started = False
    camera_fetch_completed = False
    camera_conversion_started = False
    camera_conversion_completed = False
    camera_conversion_deferred = False
    camera_fetch_deferred = False
    poststim_camera_stop_requested_utc = ""
    poststim_camera_stop_confirmed_utc = ""
    poststim_black_ended_after_fetch = False

    def defer_camera_fetch(reason, error=None):
        nonlocal camera_fetch_completed
        nonlocal camera_fetch_deferred

        camera_fetch_completed = False
        camera_fetch_deferred = True
        notes = [
            "poststim_black_active=true",
            "left_on_camera_pi=true",
            "reason=%s" % reason,
        ]
        if error is not None:
            notes.append("error=%s" % error)
        base.append_csv_row(
            event_log_path,
            {
                "event_type": "camera_fetch_deferred",
                "notes": "; ".join(notes),
            },
            base.EVENT_FIELDS,
        )

    def defer_camera_conversion(reason, error=None):
        nonlocal camera_conversion_completed
        nonlocal camera_conversion_deferred

        camera_conversion_completed = False
        camera_conversion_deferred = True
        notes = [
            "poststim_black_active=true",
            "raw_h264_available_locally=true",
            "reason=%s" % reason,
        ]
        if error is not None:
            notes.append("error=%s" % error)
        base.append_csv_row(
            event_log_path,
            {
                "event_type": "camera_conversion_deferred",
                "notes": "; ".join(notes),
            },
            base.EVENT_FIELDS,
        )

    def record_stop_request(notes):
        nonlocal poststim_camera_stop_requested_utc
        if not poststim_camera_stop_requested_utc:
            poststim_camera_stop_requested_utc = base.utc_iso_now()
        base.append_csv_row(
            event_log_path,
            {
                "event_type": "camera_stop_requested",
                "notes": notes,
            },
            base.EVENT_FIELDS,
        )

    def record_stop_confirmed(notes):
        nonlocal poststim_camera_stop_confirmed_utc
        poststim_camera_stop_confirmed_utc = base.utc_iso_now()
        base.append_csv_row(
            event_log_path,
            {
                "event_type": "camera_stop_confirmed",
                "notes": notes,
            },
            base.EVENT_FIELDS,
        )

    def resolve_camera_stop_before_exit(reason):
        nonlocal camera_stop_confirmed
        nonlocal camera_left_running_by_user

        while not camera_stop_confirmed and not camera_left_running_by_user:
            record_stop_request(reason)
            try:
                stop_camera_recording()
                camera_stop_confirmed = True
                record_stop_confirmed(reason)
                break
            except KeyboardInterrupt:
                print("ERROR stopping camera: KeyboardInterrupt while stopping camera", file=sys.stderr)
            except Exception as exc:
                print("ERROR stopping camera: %s" % exc, file=sys.stderr)

            try:
                if base.prompt_yes_no(
                    "Retry stopping the camera while keeping the screen black",
                    default_yes=True,
                ):
                    continue
            except KeyboardInterrupt:
                print("Keyboard interrupt while choosing whether to retry stopping the camera.")
                continue

            try:
                if base.prompt_yes_no(
                    "Explicitly leave the camera running and exit post-stimulus cleanup",
                    default_yes=False,
                ):
                    camera_left_running_by_user = True
                    base.append_csv_row(
                        event_log_path,
                        {
                            "event_type": "camera_left_running",
                            "notes": "reason=%s; explicitly_confirmed=true" % reason,
                        },
                        base.EVENT_FIELDS,
                    )
                    break
            except KeyboardInterrupt:
                print("Keyboard interrupt while choosing whether to leave the camera running.")
                continue

            print(
                "The camera must either be stopped or explicitly left running. "
                "The screen will remain black."
            )

    while not camera_stop_confirmed and not camera_left_running_by_user:
        try:
            should_stop = base.prompt_yes_no(
                "Stop camera recording and end the black post-stimulus baseline now",
                default_yes=False,
            )
        except KeyboardInterrupt:
            print(
                "Keyboard interrupt during black baseline. Resolving camera state while keeping the screen black."
            )
            resolve_camera_stop_before_exit("keyboard_interrupt_during_poststim_black")
            break

        if not should_stop:
            print("Black post-stimulus baseline continues. Type y and Enter when ready to stop.")
            continue

        resolve_camera_stop_before_exit("poststim_black_active=true")

    if camera_stop_confirmed and not camera_left_running_by_user:
        while True:
            camera_fetch_started = True
            base.append_csv_row(
                event_log_path,
                {
                    "event_type": "camera_fetch_started",
                    "notes": "poststim_black_active=true",
                },
                base.EVENT_FIELDS,
            )
            try:
                time.sleep(2.0)
                fetch_result = fetch_camera_recording()
                camera_fetch_completed = bool(fetch_result.get("camera_fetch_completed"))
                camera_conversion_started = True
                camera_conversion_completed = bool(fetch_result.get("camera_conversion_completed"))
                camera_conversion_deferred = bool(fetch_result.get("camera_conversion_deferred"))
                if not camera_fetch_completed:
                    raise RuntimeError(fetch_result.get("camera_fetch_error") or "Camera fetch did not complete.")
                base.append_csv_row(
                    event_log_path,
                    {
                        "event_type": "camera_fetch_completed",
                        "notes": "poststim_black_active=true; camera_conversion_completed=%s"
                        % camera_conversion_completed,
                    },
                    base.EVENT_FIELDS,
                )
                if camera_conversion_completed:
                    poststim_black_ended_after_fetch = True
                    base.append_csv_row(
                        event_log_path,
                        {
                            "event_type": "camera_conversion_completed",
                            "notes": "poststim_black_active=true; raw_h264_available_locally=true",
                        },
                        base.EVENT_FIELDS,
                    )
                    break

                camera_conversion_error = fetch_result.get("camera_conversion_error") or "MP4 conversion did not complete."
                print(
                    "Camera files were transferred, but MP4 conversion did not complete. Raw .h264 files are available on box 151.",
                    file=sys.stderr,
                )
                base.append_csv_row(
                    event_log_path,
                    {
                        "event_type": "camera_conversion_failed",
                        "notes": "poststim_black_active=true; error=%s" % camera_conversion_error,
                    },
                    base.EVENT_FIELDS,
                )

                while True:
                    try:
                        retry_conversion = base.prompt_yes_no(
                            "Retry MP4 conversion while keeping the screen black",
                            default_yes=True,
                        )
                    except KeyboardInterrupt:
                        defer_camera_conversion(
                            reason="keyboard_interrupt_at_conversion_retry_prompt",
                            error=camera_conversion_error,
                        )
                        print("MP4 conversion was interrupted; raw files remain on box 151 for later conversion.")
                        break

                    if not retry_conversion:
                        defer_camera_conversion(reason="user_declined_conversion_retry", error=camera_conversion_error)
                        print("Leaving black baseline with raw H.264 available on box 151.")
                        break

                    try:
                        time.sleep(2.0)
                        convert_result = convert_camera_recording()
                        camera_conversion_completed = bool(convert_result.get("camera_conversion_completed"))
                        camera_conversion_deferred = bool(convert_result.get("camera_conversion_deferred"))
                        if camera_conversion_completed:
                            poststim_black_ended_after_fetch = True
                            base.append_csv_row(
                                event_log_path,
                                {
                                    "event_type": "camera_conversion_completed",
                                    "notes": "poststim_black_active=true; raw_h264_available_locally=true",
                                },
                                base.EVENT_FIELDS,
                            )
                            break

                        camera_conversion_error = convert_result.get("camera_conversion_error") or "MP4 conversion did not complete."
                        print("ERROR converting camera files: %s" % camera_conversion_error, file=sys.stderr)
                        base.append_csv_row(
                            event_log_path,
                            {
                                "event_type": "camera_conversion_failed",
                                "notes": "poststim_black_active=true; error=%s" % camera_conversion_error,
                            },
                            base.EVENT_FIELDS,
                        )
                    except KeyboardInterrupt:
                        defer_camera_conversion(
                            reason="keyboard_interrupt_during_conversion",
                            error=camera_conversion_error,
                        )
                        break
                    except Exception as exc:
                        camera_conversion_error = exc
                        print("ERROR converting camera: %s" % exc, file=sys.stderr)
                        base.append_csv_row(
                            event_log_path,
                            {
                                "event_type": "camera_conversion_failed",
                                "notes": "poststim_black_active=true; error=%s" % exc,
                            },
                            base.EVENT_FIELDS,
                        )
                        try:
                            retry_conversion = base.prompt_yes_no(
                                "Retry MP4 conversion while keeping the screen black",
                                default_yes=True,
                            )
                        except KeyboardInterrupt:
                            defer_camera_conversion(
                                reason="keyboard_interrupt_at_conversion_retry_prompt",
                                error=exc,
                            )
                            break
                        if retry_conversion:
                            continue
                        defer_camera_conversion(reason="user_declined_conversion_retry", error=exc)
                        print("Leaving black baseline with raw H.264 available on box 151.")
                        break

                if camera_conversion_completed or camera_conversion_deferred:
                    break

                continue
            except KeyboardInterrupt:
                defer_camera_fetch(reason="keyboard_interrupt_during_fetch")
                print("Camera is stopped. Fetch was interrupted; files remain on the camera Pi for later retrieval.")
                break
            except Exception as exc:
                camera_fetch_completed = False
                print("ERROR fetching camera: %s" % exc, file=sys.stderr)
                base.append_csv_row(
                    event_log_path,
                    {
                        "event_type": "camera_fetch_failed",
                        "notes": "poststim_black_active=true; error=%s" % exc,
                    },
                    base.EVENT_FIELDS,
                )
                try:
                    retry_fetch = base.prompt_yes_no(
                        "Retry fetch while keeping the screen black",
                        default_yes=True,
                    )
                except KeyboardInterrupt:
                    print(
                        "Fetch retry prompt interrupted. "
                        "The camera is stopped and files remain on the camera Pi."
                    )
                    defer_camera_fetch(
                        reason="keyboard_interrupt_at_retry_prompt",
                        error=exc,
                    )
                    break

                if retry_fetch:
                    continue
                defer_camera_fetch(reason="user_declined_retry", error=exc)
                print("Leaving black baseline with camera files available for later retrieval.")
                break

    if not (camera_stop_confirmed or camera_left_running_by_user):
        raise RuntimeError("Post-stimulus cleanup invariant violated: camera state unresolved")

    poststim_black_actual_sec = time.monotonic() - poststim_black_start_monotonic
    base.append_csv_row(
        event_log_path,
        {
            "event_type": "poststim_black_end",
            "notes": "actual_sec=%.6f; camera_stop_confirmed=%s; camera_fetch_completed=%s; camera_fetch_deferred=%s; camera_left_running_by_user=%s"
            % (
                poststim_black_actual_sec,
                camera_stop_confirmed,
                camera_fetch_completed,
                camera_fetch_deferred,
                camera_left_running_by_user,
            ),
        },
        base.EVENT_FIELDS,
    )
    base.append_csv_row(
        event_log_path,
        {
            "event_type": "session_end",
            "notes": "stimulus_playback_completed=%s; camera_stop_confirmed=%s; camera_left_running_by_user=%s; camera_fetch_completed=%s; camera_conversion_completed=%s; camera_fetch_deferred=%s; camera_conversion_deferred=%s"
            % (
                stimulus_playback_completed,
                camera_stop_confirmed,
                camera_left_running_by_user,
                camera_fetch_completed,
                camera_conversion_completed,
                camera_fetch_deferred,
                camera_conversion_deferred,
            ),
        },
        base.EVENT_FIELDS,
    )

    session_completed = bool(stimulus_playback_completed and camera_stop_confirmed)
    return {
        "stimulus_playback_completed": stimulus_playback_completed,
        "poststim_gray_planned_sec": base.POSTSTIM_GRAY_PLANNED_SEC,
        "poststim_gray_includes_final_trial_iti": False,
        "poststim_gray_start_utc": poststim_gray_start_utc,
        "poststim_gray_actual_sec": poststim_gray_actual_sec,
        "poststim_black_on_utc": poststim_black_on_utc,
        "poststim_black_actual_sec": poststim_black_actual_sec,
        "poststim_camera_stop_requested_utc": poststim_camera_stop_requested_utc,
        "poststim_camera_stop_confirmed_utc": poststim_camera_stop_confirmed_utc,
        "poststim_black_ended_after_fetch": poststim_black_ended_after_fetch,
        "camera_stop_confirmed": camera_stop_confirmed,
        "camera_left_running_by_user": camera_left_running_by_user,
        "camera_fetch_started": camera_fetch_started,
        "camera_fetch_completed": camera_fetch_completed,
        "camera_conversion_started": camera_conversion_started,
        "camera_conversion_completed": camera_conversion_completed,
        "camera_conversion_deferred": camera_conversion_deferred,
        "camera_fetch_deferred": camera_fetch_deferred,
        "session_completed": session_completed,
        "poststim_screen_remained_open": True,
        "poststim_screen_remains_open_during_stop": True,
        "poststim_screen_remains_open_during_fetch": True,
        "poststim_visual_condition": "black_grayscale_0",
    }


def main():
    base.print_environment()
    try:
        import rpg
    except ImportError as exc:
        raise RuntimeError(
            "The rpg package is not installed. Install the SjulsonLab rpg repo on the behavior Pi first."
        ) from exc

    mouse_id_raw = base.prompt_text("Mouse ID: ")
    mouse_id = base.sanitize_text(mouse_id_raw) or "mouse"
    session_notes = base.prompt_text("Session notes, optional: ").strip()
    n_images_to_use = base.prompt_int_or_default("Number of unique images to use", base.N_IMAGES_TO_USE)
    n_repeats = base.prompt_int_or_default("Repeats per image", base.N_REPEATS)
    prestim_baseline_minutes = prompt_float_or_default(
        "Pre-stimulus camera baseline in minutes",
        DEFAULT_PRESTIM_BASELINE_MINUTES,
        minimum=0.0,
    )
    prestim_baseline_sec = prestim_baseline_minutes * 60.0

    image_dir = base.resolve_image_dir()
    all_pngs = base.list_png_files(image_dir)
    selected_pngs = base.select_image_subset(all_pngs, n_images_to_use)
    trials, sequence_seed = base.make_trial_sequence(selected_pngs, n_repeats)
    estimated_playback_sec = base.estimate_playback_seconds(len(trials))

    print()
    print("Session setup summary:")
    print("  Mouse ID: %s" % (mouse_id_raw.strip() or mouse_id))
    print("  Session notes, optional: %s" % session_notes)
    print("  Number of unique images: %d" % len(selected_pngs))
    print("  Repeats per image: %d" % n_repeats)
    print("  Pre-stimulus camera baseline: %s" % base.format_seconds(prestim_baseline_sec))
    print("  Camera timing: the baseline clock starts after the camera acquisition process is confirmed running")
    print("  Gray baseline: the monitor is set to the ITI-style gray frame before camera start")
    print("  Early start: type y and press Enter after recording begins")
    print("  Post-stimulus: 3.0 s controlled gray, then open-ended full-screen black")
    print("  Type y and Enter to stop the camera and end the black baseline")
    print("  The screen stays black during camera stop and file transfer")
    print("  Total trials: %d" % len(trials))
    print("  Estimated playback time: %s" % base.format_seconds(estimated_playback_sec))
    print("  Output folder root: %s" % OUTPUT_ROOT)
    print("  Session folder name: %s" % base.make_session_name(mouse_id, "YYYYMMDDThhmmssZ"))

    if not base.prompt_yes_no("Start this session", default_yes=True):
        print("Session aborted before starting. No files were changed.")
        return 0

    session_stamp = base.utc_session_stamp()
    session_id = base.make_session_name(mouse_id, session_stamp)

    session_root = base.ensure_dir(OUTPUT_ROOT / session_id)
    raw_cache_root = base.ensure_dir(session_root / "raw_cache")
    event_log_path = session_root / (session_id + "_event_log.csv")
    selected_images_path = session_root / (session_id + "_selected_images.csv")
    planned_sequence_path = session_root / (session_id + "_planned_sequence.csv")
    metadata_path = session_root / (session_id + "_metadata.json")

    print("Session ID: %s" % session_id)
    print("Selected images: %s" % selected_images_path)
    print("Planned sequence: %s" % planned_sequence_path)
    print("Event log: %s" % event_log_path)
    print("Metadata: %s" % metadata_path)

    selected_rows = []
    for index, image_path in enumerate(selected_pngs):
        image_id = base.parse_image_id_from_filename(image_path)
        selected_rows.append(
            {
                "selected_index": index,
                "image_id": image_id if image_id is not None else index,
                "image_filename": image_path.name,
                "image_path": str(image_path),
            }
        )
    base.write_csv(selected_images_path, selected_rows, ["selected_index", "image_id", "image_filename", "image_path"])

    base.write_csv(
        planned_sequence_path,
        [
            {
                "trial_index": trial["trial_index"],
                "image_index": trial["image_index"],
                "image_id": trial["image_id"],
                "image_filename": trial["image_filename"],
                "image_path": trial["image_path"],
                "repeat_number": trial["repeat_number"],
                "planned_stim_duration_sec": trial["planned_stim_duration_sec"],
                "planned_iti_duration_sec": trial["planned_iti_duration_sec"],
            }
            for trial in trials
        ],
        [
            "trial_index",
            "image_index",
            "image_id",
            "image_filename",
            "image_path",
            "repeat_number",
            "planned_stim_duration_sec",
            "planned_iti_duration_sec",
        ],
    )

    metadata = {
        "session_id": session_id,
        "utc_iso_start": base.utc_iso_now(),
        "mouse_id_input": mouse_id_raw,
        "mouse_id": mouse_id,
        "session_notes": session_notes,
        "image_dir": str(image_dir),
        "output_root": str(OUTPUT_ROOT),
        "screen_resolution": list(base.SCREEN_RESOLUTION),
        "screen_background_gray": base.SCREEN_BACKGROUND_GRAY,
        "screen_colormode": base.SCREEN_COLORMODE,
        "refresh_rate_hz": base.REFRESH_RATE_HZ,
        "n_images_to_use": n_images_to_use,
        "n_repeats": n_repeats,
        "image_subset_seed": base.IMAGE_SUBSET_SEED,
        "trial_order_seed": base.TRIAL_ORDER_SEED,
        "resolved_trial_order_seed": sequence_seed,
        "avoid_adjacent_repeats": base.AVOID_ADJACENT_REPEATS,
        "stim_duration_sec": base.STIM_DURATION_SEC,
        "iti_duration_sec": base.ITI_DURATION_SEC,
        "initial_gray_sec": base.INITIAL_GRAY_SEC,
        "final_gray_sec": base.FINAL_GRAY_SEC,
        "enable_photodiode_patch": base.ENABLE_PHOTODIODE_PATCH,
        "photodiode_size_px": base.PHOTODIODE_SIZE_PX,
        "photodiode_margin_px": base.PHOTODIODE_MARGIN_PX,
        "use_gpio": base.USE_GPIO,
        "ttl_pin_bcm": base.TTL_PIN_BCM,
        "prestim_baseline_requested_minutes": prestim_baseline_minutes,
        "prestim_baseline_requested_sec": prestim_baseline_sec,
        "prestim_early_start_enabled": True,
        "prestim_early_start_key": "y",
        "poststim_baseline_mode": "gray_transition_then_black_open_ended",
        "poststim_gray_planned_sec": base.POSTSTIM_GRAY_PLANNED_SEC,
        "poststim_gray_includes_final_trial_iti": False,
        "poststim_black_level": base.POSTSTIM_BLACK_LEVEL,
        "poststim_black_open_ended": True,
        "poststim_screen_remained_open": True,
        "poststim_screen_remains_open_during_stop": True,
        "poststim_screen_remains_open_during_fetch": True,
        "prestim_visual_condition": "gray_iti_photodiode_off",
        "prestim_screen_open_before_camera": True,
        "prestim_raw_build_under_gray": True,
        "prestim_minimum_gray_sec": base.INITIAL_GRAY_SEC,
        "prestim_baseline_clock_reference": "camera_output_growing_confirmed",
        "selected_images": selected_rows,
        "trials": trials,
        "camera_enabled": True,
        "camera_control_script": str(CAMERA_CONTROL_SCRIPT),
        "camera_stop_prompt": "Type y and Enter to stop the camera while the black baseline is active.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + chr(10))

    camera_started = False
    camera_stopped = False
    session_completed = False
    playback_started = False
    screen_gray_active = False
    raw_cache_build_duration_sec = None
    raw_cache_built_with_screen_open = False
    raw_cache_screen_compatibility_fallback = False
    camera_start_requested_utc = ""
    camera_start_returned_utc = ""
    camera_start_command_returned = False
    camera_process_confirmed = False
    camera_output_file_detected = False
    camera_output_growing_confirmed = False
    camera_output_file = ""
    camera_start_failed = False
    camera_start_error = ""
    camera_cleanup_after_start_failure_attempted = False
    camera_cleanup_after_start_failure_confirmed = False
    camera_state_unknown_acknowledged = False
    camera_start_confirmed_utc = ""
    prestim_gray_on_utc = ""
    prestim_gray_ready_utc = ""
    prestim_gray_start_monotonic = None
    prestim_baseline_start_utc = ""
    prestim_baseline_start_monotonic = None
    baseline_result = None
    poststim_result = None
    camera_stop_handled = False
    camera_fetch_completed = False
    force_start_event = None
    force_input_stop_event = None
    force_input_thread = None
    monitor_active = False
    gpio = None

    try:
        iti_raw_path = base.build_iti_raw_cache(rpg, raw_cache_root)
        if base.USE_GPIO:
            gpio = base.setup_gpio()

        try:
            with rpg.Screen(base.SCREEN_RESOLUTION, background=base.SCREEN_BACKGROUND_GRAY, colormode=base.SCREEN_COLORMODE) as screen:
                iti_raw = screen.load_raw(str(iti_raw_path))
                prestim_gray_start_monotonic = time.monotonic()
                baseline_perf, baseline_timing = base.display_raw_with_timing(screen, iti_raw)
                prestim_gray_on_utc = baseline_timing["request_utc_iso"]
                prestim_gray_ready_utc = baseline_timing["return_utc_iso"]
                screen_gray_active = True
                base.append_csv_row(
                    event_log_path,
                    {
                        "event_type": "prestim_gray_on",
                        "raw_path": str(iti_raw_path),
                        "planned_duration_sec": base.ITI_DURATION_SEC,
                        "display_request_unix_ns": baseline_timing["request_unix_ns"],
                        "display_return_unix_ns": baseline_timing["return_unix_ns"],
                        "display_return_utc_iso": baseline_timing["return_utc_iso"],
                        "display_request_perf_counter_ns": baseline_timing["request_perf_counter_ns"],
                        "display_return_perf_counter_ns": baseline_timing["return_perf_counter_ns"],
                        "display_call_duration_sec": "%.9f" % baseline_timing["duration_sec"],
                        "start_time_unix": getattr(baseline_perf, "start_time", ""),
                        "mean_interframe_us": getattr(baseline_perf, "mean_interframe", ""),
                        "stddev_interframe_us": getattr(baseline_perf, "stddev_interframe", ""),
                        "notes": "condition=gray_iti; photodiode=off; screen_opened_before_camera=true",
                    },
                    base.EVENT_FIELDS,
                )
                print("Pre-stimulus gray screen is active.")
                sys.stdout.flush()

                camera_start_requested_utc = base.utc_iso_now()
                base.append_csv_row(
                    event_log_path,
                    {
                        "event_type": "camera_start_requested",
                        "notes": "gray_already_active=true; prestim_baseline_requested_sec=%.3f" % prestim_baseline_sec,
                    },
                    base.EVENT_FIELDS,
                )
                camera_start_result = start_camera_recording_with_recovery(
                    mouse_id,
                    session_id,
                    event_log_path,
                )
                camera_controller_state = camera_start_result["controller_state"]
                camera_start_command_returned = bool(
                    camera_controller_state.get("camera_start_command_returned")
                )
                camera_process_confirmed = bool(camera_controller_state.get("camera_process_confirmed"))
                camera_output_file_detected = bool(camera_controller_state.get("camera_output_file_detected"))
                camera_output_growing_confirmed = bool(
                    camera_controller_state.get("camera_output_growing_confirmed")
                )
                camera_output_file = camera_controller_state.get("camera_output_file") or ""
                camera_start_failed = not camera_start_result["confirmed_running"]
                camera_start_error = camera_start_result["error"]
                camera_cleanup_after_start_failure_attempted = camera_start_result["cleanup_attempted"]
                camera_cleanup_after_start_failure_confirmed = camera_start_result["cleanup_confirmed"]
                camera_state_unknown_acknowledged = camera_start_result["camera_state_unknown_acknowledged"]
                if not camera_start_result["confirmed_running"]:
                    raise RuntimeError(
                        "Experiment aborted because camera recording was not confirmed: %s"
                        % camera_start_error
                    )

                camera_started = True
                camera_start_returned_utc = base.utc_iso_now()
                camera_start_confirmed_utc = (
                    camera_controller_state.get("camera_output_growth_confirmed_utc")
                    or camera_start_returned_utc
                )
                prestim_baseline_start_utc = camera_start_confirmed_utc
                prestim_baseline_start_monotonic = time.monotonic()
                base.append_csv_row(
                    event_log_path,
                    {
                        "event_type": "camera_recording_confirmed",
                        "notes": "camera_pid_confirmed=true; camera_output_growing_confirmed=true; gray_already_active=true",
                    },
                    base.EVENT_FIELDS,
                )
                base.append_csv_row(
                    event_log_path,
                    {
                        "event_type": "prestim_baseline_start",
                        "notes": "requested_sec=%.3f; gray_already_active=true; camera_output_growing_confirmed=true" % prestim_baseline_sec,
                    },
                    base.EVENT_FIELDS,
                )

                force_start_event, force_input_stop_event, force_input_thread, monitor_active = start_prestimulus_early_start_monitor()
                print()
                print("Remote camera recording is active.")
                print("Pre-stimulus baseline target: %s." % base.format_seconds(prestim_baseline_sec))
                print("The requested baseline clock begins after recording is confirmed.")
                print("Stimulus raw files will be prepared while gray remains on the screen.")
                print("Type y and press Enter to request early stimulus start.")
                sys.stdout.flush()

                print("Preparing session raw files...")
                raw_cache_build_start = time.perf_counter()
                stim_raw_paths = base.build_stim_raw_cache(rpg, raw_cache_root, selected_pngs)
                raw_cache_build_duration_sec = time.perf_counter() - raw_cache_build_start
                raw_cache_built_with_screen_open = True
                baseline_elapsed_after_raw = time.monotonic() - prestim_baseline_start_monotonic
                gray_elapsed_after_raw = time.monotonic() - prestim_gray_start_monotonic
                baseline_remaining_after_raw = max(0.0, prestim_baseline_sec - baseline_elapsed_after_raw)
                base.append_csv_row(
                    event_log_path,
                    {
                        "event_type": "raw_cache_ready",
                        "notes": "build_duration_sec=%.6f; baseline_elapsed_sec=%.6f; gray_elapsed_sec=%.6f; gray_active=true"
                        % (
                            raw_cache_build_duration_sec,
                            baseline_elapsed_after_raw,
                            gray_elapsed_after_raw,
                        ),
                    },
                    base.EVENT_FIELDS,
                )
                print("Session raw files are ready.")
                print(
                    "Camera baseline elapsed: %s; remaining: %s."
                    % (base.format_seconds(baseline_elapsed_after_raw), base.format_seconds(baseline_remaining_after_raw))
                )
                print("Type y and press Enter to start early.")
                sys.stdout.flush()

                loaded_stim_raws = {}
                for image_path in selected_pngs:
                    loaded_stim_raws[image_path.stem] = screen.load_raw(str(stim_raw_paths[image_path.stem]))

                baseline_result = wait_for_prestimulus_gate(
                    prestim_baseline_start_monotonic,
                    prestim_baseline_sec,
                    prestim_gray_start_monotonic,
                    base.INITIAL_GRAY_SEC,
                    force_start_event,
                )
                stop_prestimulus_early_start_monitor(force_input_stop_event, force_input_thread)
                force_input_thread = None
                force_input_stop_event = None
                monitor_active = False

                base.append_csv_row(
                    event_log_path,
                    {
                        "event_type": "prestim_baseline_end",
                        "notes": "reason=%s; requested_sec=%.3f; camera_elapsed_sec=%.3f; gray_elapsed_sec=%.3f; forced=%s; waited_for_minimum_gray_after_override=%s"
                        % (
                            baseline_result["end_reason"],
                            baseline_result["requested_sec"],
                            baseline_result["camera_baseline_elapsed_sec"],
                            baseline_result["gray_elapsed_sec"],
                            baseline_result["forced"],
                            baseline_result["waited_for_minimum_gray_after_override"],
                        ),
                    },
                    base.EVENT_FIELDS,
                )
                base.append_csv_row(
                    event_log_path,
                    {
                        "event_type": "session_start",
                        "notes": "stimulus_playback_begin; screen_already_open=true",
                    },
                    base.EVENT_FIELDS,
                )
                print(prestimulus_result_message(baseline_result))
                print("Stimulus playback is now active.")
                sys.stdout.flush()

                playback_started = True
                base.run_trial_sequence(
                    screen,
                    trials,
                    loaded_stim_raws,
                    iti_raw,
                    stim_raw_paths,
                    iti_raw_path,
                    event_log_path,
                    gpio=gpio if base.USE_GPIO else None,
                    include_final_iti=False,
                )

                base.append_csv_row(
                    event_log_path,
                    {
                        "event_type": "stimulus_playback_end",
                        "notes": "last_trial_completed=true",
                    },
                    base.EVENT_FIELDS,
                )
                poststim_result = run_poststim_black_baseline(screen, iti_raw, event_log_path)
                camera_stop_handled = poststim_result["camera_stop_confirmed"] or poststim_result["camera_left_running_by_user"]
                camera_fetch_completed = poststim_result["camera_fetch_completed"]
                session_completed = poststim_result["session_completed"]

        except KeyboardInterrupt:
            stop_prestimulus_early_start_monitor(force_input_stop_event, force_input_thread)
            force_input_thread = None
            force_input_stop_event = None
            monitor_active = False
            if not playback_started:
                base.append_csv_row(
                    event_log_path,
                    {
                        "event_type": "session_end",
                        "notes": "keyboard_interrupt_during_prestim",
                    },
                    base.EVENT_FIELDS,
                )
            else:
                base.append_csv_row(
                    event_log_path,
                    {
                        "event_type": "session_end",
                        "notes": "keyboard_interrupt",
                    },
                    base.EVENT_FIELDS,
                )
            raise
        except Exception as exc:
            stop_prestimulus_early_start_monitor(force_input_stop_event, force_input_thread)
            force_input_thread = None
            force_input_stop_event = None
            monitor_active = False
            if camera_started and not playback_started:
                base.append_csv_row(
                    event_log_path,
                    {
                        "event_type": "session_end",
                        "notes": "raw_cache_failure_during_prestim; error=%s" % exc,
                    },
                    base.EVENT_FIELDS,
                )
            raise
        finally:
            if base.USE_GPIO and gpio is not None:
                try:
                    import RPi.GPIO as GPIO

                    GPIO.output(base.TTL_PIN_BCM, GPIO.LOW)
                    GPIO.cleanup()
                except Exception:
                    pass

        return 0

    finally:
        stop_prestimulus_early_start_monitor(force_input_stop_event, force_input_thread)
        camera_stopped = poststim_result["camera_stop_confirmed"] if poststim_result else False
        camera_fetch_completed = poststim_result["camera_fetch_completed"] if poststim_result else False
        if camera_started and not camera_stop_handled:
            if base.prompt_yes_no("Stop camera recording now", default_yes=False):
                try:
                    stop_camera_recording()
                    camera_stopped = True
                except Exception as exc:
                    print("ERROR stopping camera: %s" % exc, file=sys.stderr)
            else:
                print(
                    "Camera left running. Stop it later with: "
                    "python3 remote_camera_control.py stop"
                )

        metadata["utc_iso_end"] = base.utc_iso_now()
        metadata["event_log"] = str(event_log_path)
        metadata["selected_images_csv"] = str(selected_images_path)
        metadata["planned_sequence_csv"] = str(planned_sequence_path)
        metadata["raw_cache_root"] = str(raw_cache_root)
        metadata["camera_started"] = camera_started
        metadata["camera_stopped"] = camera_stopped
        metadata["camera_fetch_completed"] = camera_fetch_completed
        metadata["camera_conversion_started"] = poststim_result["camera_conversion_started"] if poststim_result else False
        metadata["camera_conversion_completed"] = poststim_result["camera_conversion_completed"] if poststim_result else False
        metadata["camera_conversion_deferred"] = poststim_result["camera_conversion_deferred"] if poststim_result else False
        metadata["camera_fetch_deferred"] = poststim_result["camera_fetch_deferred"] if poststim_result else False
        metadata["camera_left_running_by_user"] = poststim_result["camera_left_running_by_user"] if poststim_result else False
        metadata["camera_fetch_started"] = poststim_result["camera_fetch_started"] if poststim_result else False
        metadata["camera_session_id"] = session_id if camera_started else ""
        metadata["camera_start_requested_utc"] = camera_start_requested_utc
        metadata["camera_start_returned_utc"] = camera_start_returned_utc
        metadata["camera_start_confirmed_utc"] = camera_start_confirmed_utc
        metadata["camera_start_command_returned"] = camera_start_command_returned
        metadata["camera_process_confirmed"] = camera_process_confirmed
        metadata["camera_pid_confirmed"] = camera_process_confirmed
        metadata["camera_output_file_detected"] = camera_output_file_detected
        metadata["camera_output_growing_confirmed"] = camera_output_growing_confirmed
        metadata["camera_output_file"] = camera_output_file
        metadata["camera_readiness_reference"] = "pid_alive_and_output_size_increasing"
        metadata["camera_start_failed"] = camera_start_failed
        metadata["camera_start_error"] = camera_start_error
        metadata["camera_cleanup_after_start_failure_attempted"] = camera_cleanup_after_start_failure_attempted
        metadata["camera_cleanup_after_start_failure_confirmed"] = camera_cleanup_after_start_failure_confirmed
        metadata["camera_state_unknown_acknowledged"] = camera_state_unknown_acknowledged
        metadata["session_completed"] = session_completed
        metadata["playback_started"] = playback_started
        metadata["poststim_baseline_mode"] = "gray_transition_then_black_open_ended"
        metadata["poststim_gray_planned_sec"] = base.POSTSTIM_GRAY_PLANNED_SEC
        metadata["poststim_gray_includes_final_trial_iti"] = False
        metadata["poststim_black_level"] = base.POSTSTIM_BLACK_LEVEL
        metadata["poststim_black_open_ended"] = True
        metadata["poststim_visual_condition"] = poststim_result["poststim_visual_condition"] if poststim_result else ""
        metadata["poststim_screen_remained_open"] = poststim_result["poststim_screen_remained_open"] if poststim_result else False
        metadata["poststim_screen_remains_open_during_stop"] = poststim_result["poststim_screen_remains_open_during_stop"] if poststim_result else False
        metadata["poststim_screen_remains_open_during_fetch"] = poststim_result["poststim_screen_remains_open_during_fetch"] if poststim_result else False
        metadata["poststim_black_on_utc"] = poststim_result["poststim_black_on_utc"] if poststim_result else ""
        metadata["poststim_camera_stop_requested_utc"] = poststim_result["poststim_camera_stop_requested_utc"] if poststim_result else ""
        metadata["poststim_camera_stop_confirmed_utc"] = poststim_result["poststim_camera_stop_confirmed_utc"] if poststim_result else ""
        metadata["poststim_gray_start_utc"] = poststim_result["poststim_gray_start_utc"] if poststim_result else ""
        metadata["poststim_gray_actual_sec"] = poststim_result["poststim_gray_actual_sec"] if poststim_result else ""
        metadata["poststim_black_actual_sec"] = poststim_result["poststim_black_actual_sec"] if poststim_result else ""
        metadata["poststim_black_ended_after_fetch"] = poststim_result["poststim_black_ended_after_fetch"] if poststim_result else False
        metadata["prestim_visual_condition"] = "gray_iti_photodiode_off" if screen_gray_active else ""
        metadata["prestim_screen_open_before_camera"] = True
        metadata["prestim_raw_build_under_gray"] = raw_cache_built_with_screen_open
        metadata["prestim_minimum_gray_sec"] = base.INITIAL_GRAY_SEC
        metadata["prestim_baseline_clock_reference"] = "camera_output_growing_confirmed"
        metadata["prestim_gray_on_utc"] = prestim_gray_on_utc
        metadata["prestim_gray_ready_utc"] = prestim_gray_ready_utc
        metadata["prestim_gray_before_camera_start_sec"] = (
            prestim_baseline_start_monotonic - prestim_gray_start_monotonic
            if prestim_baseline_start_monotonic is not None and prestim_gray_start_monotonic is not None
            else ""
        )
        metadata["prestim_camera_baseline_actual_sec"] = (
            baseline_result["camera_baseline_elapsed_sec"] if baseline_result else ""
        )
        metadata["prestim_gray_actual_sec_before_first_stim"] = (
            baseline_result["gray_elapsed_sec"] if baseline_result else ""
        )
        metadata["prestim_minimum_gray_satisfied"] = (
            baseline_result["gray_elapsed_sec"] >= base.INITIAL_GRAY_SEC if baseline_result else False
        )
        metadata["prestim_baseline_forced"] = baseline_result["forced"] if baseline_result else False
        metadata["prestim_baseline_end_reason"] = baseline_result["end_reason"] if baseline_result else ""
        metadata["prestim_baseline_remaining_sec_at_gate_entry"] = (
            baseline_result["remaining_camera_baseline_sec_at_gate_entry"] if baseline_result else ""
        )
        metadata["raw_cache_build_duration_sec"] = raw_cache_build_duration_sec if raw_cache_build_duration_sec is not None else ""
        metadata["raw_cache_built_with_screen_open"] = raw_cache_built_with_screen_open
        metadata["raw_cache_screen_compatibility_fallback"] = raw_cache_screen_compatibility_fallback
        metadata["prestim_gate_released_utc"] = base.utc_iso_now() if baseline_result else ""
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + chr(10))
        if session_completed:
            print("Session finished. Files are in: %s" % session_root)
        else:
            print("Session stopped early. Partial files are in: %s" % session_root)
        sys.stdout.flush()


if __name__ == "__main__":
    print(
        "This repository's supported experiment entrypoint is:\n\n"
        "  python3 run_stringer_reward_conditioning.py\n\n"
        "For the natural-image-only protocol use:\n"
        "  https://github.com/hung-lo/vstim_natural",
        file=sys.stderr,
    )
    raise SystemExit(2)
