#!/usr/bin/env python3
"""Runtime regression tests for reward-conditioning behavior."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_stringer_reward_conditioning as reward


class FakeGPIOClient:
    def __init__(self):
        self.contexts = []
        self.drain_calls = 0

    def set_context(self, context):
        self.contexts.append(dict(context))
        return {"command": "set_context"}

    def drain_events(self):
        self.drain_calls += 1
        return []

    def is_alive(self):
        return True


class RewardConditioningRuntimeTests(unittest.TestCase):
    def test_build_trial_summary_uses_monotonic_lick_timestamps(self):
        trials = [
            {
                "trial_index": 0,
                "trial_number": 1,
                "block_number": 1,
                "image_role": "low_01",
                "image_category": "low_probability_unrewarded",
                "image_id": 1,
                "image_filename": "img_0001.png",
                "reward_eligible": False,
                "reward_scheduled": False,
                "reward_omission_scheduled": False,
                "planned_iti_duration_sec": 0.25,
            },
            {
                "trial_index": 1,
                "trial_number": 2,
                "block_number": 1,
                "image_role": "low_02",
                "image_category": "low_probability_unrewarded",
                "image_id": 2,
                "image_filename": "img_0002.png",
                "reward_eligible": False,
                "reward_scheduled": False,
                "reward_omission_scheduled": False,
                "planned_iti_duration_sec": 0.25,
            },
        ]
        runtime_by_trial = {
            0: {
                "trial_executed": True,
                "stim_presented": True,
                "trial_completed": True,
                "stim_request_monotonic_ns": 1_000_000_000,
                "stim_request_unix_ns": 1_000_000_000,
                "stim_segment1_return_unix_ns": 1_050_000_000,
                "stim_segment2_request_unix_ns": 1_100_000_000,
                "stim_offset_request_unix_ns": 1_150_000_000,
                "stim_offset_monotonic_ns": 1_150_000_000,
                "segment_boundary_gap_sec": 0.002,
                "reward_command_id": "",
            },
            1: {
                "trial_executed": True,
                "stim_presented": True,
                "trial_completed": True,
                "stim_request_monotonic_ns": 2_000_000_000,
                "stim_request_unix_ns": 2_000_000_000,
                "stim_segment1_return_unix_ns": 2_050_000_000,
                "stim_segment2_request_unix_ns": 2_100_000_000,
                "stim_offset_request_unix_ns": 2_150_000_000,
                "stim_offset_monotonic_ns": 2_150_000_000,
                "segment_boundary_gap_sec": 0.002,
                "reward_command_id": "",
            },
        }
        all_gpio_events = [
            {
                "event_type": "lick_onset",
                "monotonic_ns": 1_750_000_000,
                "unix_time_ns": 1_750_000_000,
                "trial_index": 0,
            }
        ]

        rows = reward.build_trial_summary(trials, runtime_by_trial, all_gpio_events)
        self.assertEqual(rows[0]["lick_count_pre_0p5_sec"], 0)
        self.assertEqual(rows[1]["lick_count_pre_0p5_sec"], 1)
        self.assertEqual(rows[1]["trial_executed"], True)
        self.assertEqual(rows[1]["trial_completed"], True)

    def test_zero_post_services_final_rewarded_suction(self):
        gpio_client = FakeGPIOClient()
        gpio_client.trigger_suction = mock.Mock(return_value="suction_1")
        pending = [{
            "command_id": "reward_1",
            "expected_num_pulses": 6,
            "timeout_sec": 0.5,
        }, {
            "suction_target": 0.0,
            "duration_sec": 0.01,
            "context": {"phase": "stimulus", "suction_scheduled": True},
            "runtime": {},
        }]
        with tempfile.TemporaryDirectory(prefix="reward_post_zero_") as temp_dir:
            with mock.patch.object(reward, "wait_until"), \
                 mock.patch.object(reward, "verify_reward_command") as verify_reward, \
                 mock.patch.object(reward, "verify_suction_command") as verify_suction:
                reward.hold_background(
                    "poststim_background", 0.0, gpio_client,
                    Path(temp_dir) / "events.csv", [], pending_reward_checks=pending
                )
        verify_reward.assert_called_once()
        verify_suction.assert_called_once()
        gpio_client.trigger_suction.assert_called_once()

    def test_zero_post_services_final_omission_suction(self):
        gpio_client = FakeGPIOClient()
        gpio_client.trigger_suction = mock.Mock(return_value="suction_omission_1")
        pending = [{
            "suction_target": 0.0,
            "duration_sec": 0.01,
            "context": {"phase": "stimulus", "reward_scheduled": False, "suction_scheduled": True},
            "runtime": {},
        }]
        with tempfile.TemporaryDirectory(prefix="reward_post_zero_omission_") as temp_dir:
            with mock.patch.object(reward, "wait_until"), \
                 mock.patch.object(reward, "verify_suction_command") as verify_suction:
                reward.hold_background(
                    "poststim_background", 0.0, gpio_client,
                    Path(temp_dir) / "events.csv", [], pending_reward_checks=pending
                )
        verify_suction.assert_called_once()
        gpio_client.trigger_suction.assert_called_once()

    def test_qc_detects_rogue_commands_but_ignores_manual_commands(self):
        trial = {
            "trial_index": 0, "trial_number": 1, "block_number": 1,
            "image_role": "low_01", "image_category": "low_probability_unrewarded",
            "image_id": 1, "image_filename": "img.png",
            "reward_eligible": False, "reward_scheduled": False,
            "reward_omission_scheduled": False, "suction_scheduled": False,
            "planned_iti_duration_sec": 3.0,
        }
        row = reward._blank_trial_summary(trial)
        row.update({"trial_executed": True, "stim_presented": True, "trial_completed": True})
        base_events = [
            {"event_type": "manual_reward_command_received", "command_id": "manual_reward_1"},
            {"event_type": "suction_command_received", "command_id": "manual_suction_1", "phase": "manual_suction"},
        ]
        valid = reward.build_session_qc("session", [trial], [row], base_events, 6)
        self.assertTrue(valid["qc_pass"])

        rogue = base_events + [
            {"event_type": "reward_command_received", "command_id": "rogue_reward", "phase": "iti"},
            {"event_type": "suction_command_received", "command_id": "rogue_suction", "phase": "iti"},
        ]
        invalid = reward.build_session_qc("session", [trial], [row], rogue, 6)
        self.assertEqual(invalid["unexpected_reward_command_ids"], ["rogue_reward"])
        self.assertEqual(invalid["unexpected_suction_command_ids"], ["rogue_suction"])
        self.assertFalse(invalid["qc_pass"])

    def test_run_trials_skips_final_iti(self):
        trials = [
            {
                "trial_index": 0,
                "trial_number": 1,
                "block_number": 1,
                "image_role": "low_01",
                "image_category": "low_probability_unrewarded",
                "image_id": 1,
                "image_filename": "img_0001.png",
                "reward_eligible": False,
                "reward_scheduled": False,
                "reward_omission_scheduled": False,
                "planned_iti_duration_sec": 0.25,
            },
            {
                "trial_index": 1,
                "trial_number": 2,
                "block_number": 1,
                "image_role": "low_02",
                "image_category": "low_probability_unrewarded",
                "image_id": 2,
                "image_filename": "img_0002.png",
                "reward_eligible": False,
                "reward_scheduled": False,
                "reward_omission_scheduled": False,
                "planned_iti_duration_sec": 0.25,
            },
        ]
        loaded_raws = {
            "img_0001.png": {"first": Path("first1.raw"), "second": Path("second1.raw")},
            "img_0002.png": {"first": Path("first2.raw"), "second": Path("second2.raw")},
        }
        raw_paths = {
            "img_0001.png": {"first": Path("first1.raw"), "second": Path("second1.raw")},
            "img_0002.png": {"first": Path("first2.raw"), "second": Path("second2.raw")},
        }
        fake_screen = SimpleNamespace()
        gpio_client = FakeGPIOClient()
        call_counter = {"count": 0}
        wait_calls = []

        def fake_display_raw_with_timing(screen, raw_path):
            call_index = call_counter["count"]
            call_counter["count"] += 1
            base_ns = 1_000_000_000 * (call_index + 1)
            perf = SimpleNamespace(start_time=base_ns / 1_000_000_000.0, mean_interframe=0.0, stddev_interframe=0.0)
            timing = {
                "request_utc_iso": f"iso-{call_index}",
                "request_unix_sec": f"{base_ns / 1_000_000_000.0:.9f}",
                "request_unix_ns": base_ns,
                "request_perf_counter_ns": base_ns + 100,
                "return_unix_ns": base_ns + 50_000,
                "return_perf_counter_ns": base_ns + 200,
                "duration_sec": 0.001,
            }
            return perf, timing

        def fake_wait_until(deadline_monotonic, gpio_client_arg, event_log_path, all_gpio_events):
            wait_calls.append(deadline_monotonic)

        with tempfile.TemporaryDirectory(prefix="reward_runtime_test_") as temp_dir:
            event_log_path = Path(temp_dir) / "event_log.csv"
            background_raw_path = Path(temp_dir) / "gray.raw"
            loaded_background_raw = Path(temp_dir) / "gray_loaded.raw"
            with mock.patch.object(reward.base, "display_raw_with_timing", side_effect=fake_display_raw_with_timing),                  mock.patch.object(reward, "wait_until", side_effect=fake_wait_until):
                runtime_by_trial, pending_final_reward_checks = reward.run_trials(
                    fake_screen,
                    trials,
                    loaded_raws,
                    loaded_background_raw,
                    raw_paths,
                    background_raw_path,
                    gpio_client,
                    event_log_path,
                    [],
                    reward_num_pulses=6,
                    reward_verification_timeout_sec=0.5,
                    suction_delay_sec=3.5,
                    suction_duration_sec=0.05,
                )

        self.assertEqual(len(wait_calls), 1)
        self.assertTrue(runtime_by_trial[0]["trial_completed"])
        self.assertTrue(runtime_by_trial[1]["trial_completed"])
        self.assertEqual(pending_final_reward_checks, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
