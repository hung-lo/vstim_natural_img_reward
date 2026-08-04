#!/usr/bin/env python3
"""Simulation test proving reward delivery is not contingent on a lick."""

import time

from reward_conditioning_gpio import BehaviorGPIOClient


def main():
    config = {
        "simulate_gpio": True,
        "reward_pin_bcm": 19,
        "lick_pin_bcm": 26,
        "lick_bounce_time_sec": None,
        "reward_pulse_on_sec": 0.002,
        "reward_pulse_off_sec": 0.001,
        "reward_num_pulses": 3,
    }
    client = BehaviorGPIOClient(config)
    client.start()
    context = {
        "phase": "stimulus",
        "trial_index": 0,
        "trial_number": 1,
        "block_number": 1,
        "image_role": "rewarded_high_1",
        "image_filename": "example.png",
        "reward_scheduled": True,
    }
    client.set_context(context)

    # No lick event is generated or supplied.  The precomputed reward command
    # alone must create the full valve pulse train.
    client.trigger_reward(context)
    time.sleep(0.05)
    events = client.drain_events()
    events.extend(client.shutdown())

    event_types = [event.get("event_type") for event in events]
    valve_on_count = event_types.count("reward_valve_on")
    assert "reward_command_received" in event_types
    assert valve_on_count == config["reward_num_pulses"]
    assert "reward_complete" in event_types
    assert "lick_onset" not in event_types
    print("PASS: reward pulse train completed with no lick event.")


if __name__ == "__main__":
    main()
