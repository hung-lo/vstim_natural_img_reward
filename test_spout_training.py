#!/usr/bin/env python3
"""Tests for the gray-screen spout-training protocol."""

import unittest

import run_spout_training as training


class SpoutTrainingTests(unittest.TestCase):
    def test_default_schedule_is_60_and_intervals_are_in_range(self):
        schedule = training.build_training_schedule(
            1_000_000_000, seed=1234,
        )
        self.assertEqual(len(schedule), 60)
        self.assertEqual(schedule[0]["planned_interval_sec"], 0.0)
        intervals = [row["planned_interval_sec"] for row in schedule[1:]]
        self.assertTrue(all(8.0 <= value <= 12.0 for value in intervals))

    def test_schedule_is_deterministic_for_seed(self):
        self.assertEqual(
            training.build_training_schedule(100, 5, seed=42),
            training.build_training_schedule(100, 5, seed=42),
        )

    def test_schedule_can_be_anchored_after_baiting(self):
        intervals = training.build_training_intervals(3, seed=42)
        schedule = training.anchor_training_schedule(50_000_000_000, intervals)
        self.assertEqual(schedule[0]["planned_reward_target_monotonic_ns"], 50_000_000_000)
        self.assertEqual(
            schedule[1]["planned_reward_target_monotonic_ns"]
            - schedule[0]["planned_reward_target_monotonic_ns"],
            round(intervals[1] * 1e9),
        )

    def test_criterion_boundaries_and_recent_window(self):
        self.assertFalse(training.evaluate_training_criterion([True] * 19)["criterion_evaluable"])
        self.assertTrue(training.evaluate_training_criterion([True] * 16 + [False] * 4)["criterion_passed"])
        self.assertFalse(training.evaluate_training_criterion([True] * 15 + [False] * 5)["criterion_passed"])
        result = training.evaluate_training_criterion(
            [True] * 20 + [False] * 4 + [True] * 16
        )
        self.assertEqual(result["recent_success_count"], 16)
        self.assertEqual(result["recent_success_fraction"], 0.8)

    def test_bait_is_not_part_of_training_criterion(self):
        result = training.evaluate_training_criterion([True] * 16 + [False] * 4)
        self.assertEqual(result["recent_success_count"], 16)

    def test_lick_metrics_use_reward_and_suction_timestamps(self):
        events = [
            {"event_type": "lick_onset", "monotonic_ns": 9_500_000_000},
            {"event_type": "lick_onset", "monotonic_ns": 10_600_000_000},
            {"event_type": "lick_onset", "monotonic_ns": 11_500_000_000},
            {"event_type": "lick_onset", "monotonic_ns": 13_500_000_000},
        ]
        metrics = training.compute_reward_lick_metrics(
            events, 10_000_000_000, 12_500_000_000,
        )
        self.assertEqual(metrics["lick_count_pre_reward_1p0_sec"], 1)
        self.assertEqual(metrics["lick_count_reward_to_0p5_sec"], 0)
        self.assertEqual(metrics["lick_count_reward_to_1p0_sec"], 1)
        self.assertEqual(metrics["lick_count_reward_to_2p5_sec"], 2)
        self.assertEqual(metrics["first_lick_latency_sec"], 0.6)
        self.assertTrue(metrics["retrieval_success"])

    def test_late_or_pre_reward_only_lick_does_not_retrieve(self):
        metrics = training.compute_reward_lick_metrics(
            [{"event_type": "lick_onset", "monotonic_ns": 9_500_000_000},
             {"event_type": "lick_onset", "monotonic_ns": 13_500_000_000}],
            10_000_000_000, 12_500_000_000,
        )
        self.assertFalse(metrics["retrieval_success"])
        self.assertIsNone(metrics["first_lick_latency_sec"])

    def test_qc_keeps_training_outcome_separate(self):
        rows = [{"reward_command_id": "r1", "suction_command_id": "s1"}]
        events = [
            {"command_id": "r1", "event_type": "reward_command_received"},
            {"command_id": "r1", "event_type": "reward_complete"},
            {"command_id": "r1", "event_type": "reward_valve_on"},
            {"command_id": "r1", "event_type": "reward_valve_off"},
            {"command_id": "s1", "event_type": "suction_command_received"},
            {"command_id": "s1", "event_type": "suction_on"},
            {"command_id": "s1", "event_type": "suction_off"},
            {"command_id": "s1", "event_type": "suction_complete"},
        ]
        qc = training.build_training_qc(
            rows, events, False,
            {"window": 20, "recent_success_fraction": 0.0},
        )
        self.assertTrue(qc["qc_pass"])
        self.assertFalse(qc["training_passed"])

    def test_qc_excludes_bait_commands(self):
        rows = [{"reward_command_id": "r1", "suction_command_id": "s1"}]
        events = [
            {"phase": "spout_training_bait", "command_id": "bait-r",
             "event_type": "reward_command_received"},
            {"phase": "spout_training_bait", "command_id": "bait-r",
             "event_type": "reward_complete"},
            {"phase": "spout_training_bait", "command_id": "bait-s",
             "event_type": "suction_complete"},
            {"phase": "spout_training", "command_id": "r1",
             "event_type": "reward_command_received"},
            {"phase": "spout_training", "command_id": "r1",
             "event_type": "reward_complete"},
            {"phase": "spout_training", "command_id": "r1",
             "event_type": "reward_valve_on"},
            {"phase": "spout_training", "command_id": "r1",
             "event_type": "reward_valve_off"},
            {"phase": "spout_training", "command_id": "s1",
             "event_type": "suction_command_received"},
            {"phase": "spout_training", "command_id": "s1",
             "event_type": "suction_on"},
            {"phase": "spout_training", "command_id": "s1",
             "event_type": "suction_off"},
            {"phase": "spout_training", "command_id": "s1",
             "event_type": "suction_complete"},
        ]
        qc = training.build_training_qc(
            rows, events, False,
            {"window": 20, "recent_success_fraction": 0.0},
        )
        self.assertTrue(qc["qc_pass"])
        self.assertEqual(qc["unexpected_reward_command_ids"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
