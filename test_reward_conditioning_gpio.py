#!/usr/bin/env python3
"""Tests for the simulation-only GPIO helpers."""

import time
import unittest
from unittest import mock

from reward_conditioning_gpio import BehaviorGPIOClient


def _simulation_config():
    return {
        "simulate_gpio": True,
        "reward_pin_bcm": 19,
        "suction_pin_bcm": 25,
        "lick_pin_bcm": 26,
        "suction_duration_sec": 0.01,
        "lick_bounce_time_sec": None,
        "reward_pulse_on_sec": 0.002,
        "reward_pulse_off_sec": 0.001,
        "reward_num_pulses": 3,
    }


class RewardConditioningGPIOTests(unittest.TestCase):
    def test_simulate_lick_emits_contextual_synthetic_onset(self):
        client = BehaviorGPIOClient(_simulation_config())
        context = {
            "phase": "iti",
            "trial_index": 4,
            "trial_number": 5,
            "block_number": 1,
            "image_role": "rewarded_high_1",
            "image_filename": "example.png",
            "reward_scheduled": True,
            "suction_scheduled": True,
        }
        try:
            client.start()
            client.set_context(context)
            client.simulate_lick()
            deadline = time.monotonic() + 1.0
            events = []
            while time.monotonic() < deadline:
                events.extend(client.drain_events())
                if any(event.get("event_type") == "lick_onset" for event in events):
                    break
                time.sleep(0.01)
            lick = next(event for event in events if event.get("event_type") == "lick_onset")
            self.assertEqual(lick["lick_edge"], "simulated_test")
            self.assertEqual(lick["notes"], "synthetic_water_accounting_validation")
            for key, value in context.items():
                self.assertEqual(lick[key], value)
            self.assertIn("monotonic_ns", lick)
        finally:
            client.shutdown()

    def test_simulate_lick_refuses_real_gpio_before_send(self):
        config = dict(_simulation_config(), simulate_gpio=False)
        client = BehaviorGPIOClient(config)
        client._connection = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "real GPIO mode"):
            client.simulate_lick()
        client._connection.send.assert_not_called()

    def test_schedule_simulated_lick_emits_asynchronously(self):
        client = BehaviorGPIOClient(_simulation_config())
        try:
            client.start()
            client.schedule_simulated_lick(0.05)
            immediate = client.drain_events()
            self.assertFalse(
                any(event.get("lick_edge") == "simulated_behavior_test" for event in immediate)
            )
            deadline = time.monotonic() + 1.0
            events = []
            while time.monotonic() < deadline:
                events.extend(client.drain_events())
                if any(event.get("lick_edge") == "simulated_behavior_test" for event in events):
                    break
                time.sleep(0.01)
            lick = next(
                event for event in events
                if event.get("lick_edge") == "simulated_behavior_test"
            )
            self.assertIn("monotonic_ns", lick)
            self.assertEqual(
                lick["notes"], "synthetic_anticipatory_behavior_validation"
            )
        finally:
            client.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
