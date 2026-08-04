#!/usr/bin/env python3
"""Hardware-free regression tests for the reward-conditioning planner."""

import shutil
import tempfile
import unittest
from pathlib import Path

import reward_conditioning_protocol as protocol


class RewardConditioningProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="reward_conditioning_test_"))
        self.images = [Path("natural_image_%04d.png" % index) for index in range(100)]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_shared_panel_and_fixed_mouse_assignment(self):
        mouse_a_1, path_a, created_a_1, _ = protocol.create_or_load_assignment(
            "MOUSE_A", self.images, self.temp_dir
        )
        mouse_a_2, path_a_2, created_a_2, _ = protocol.create_or_load_assignment(
            "MOUSE_A", self.images, self.temp_dir
        )
        mouse_b, _, _, _ = protocol.create_or_load_assignment(
            "MOUSE_B", self.images, self.temp_dir
        )

        self.assertTrue(created_a_1)
        self.assertFalse(created_a_2)
        self.assertEqual(path_a, path_a_2)
        self.assertEqual(
            [row["image_filename"] for row in mouse_a_1],
            [row["image_filename"] for row in mouse_a_2],
        )
        self.assertEqual(
            {row["image_filename"] for row in mouse_a_1},
            {row["image_filename"] for row in mouse_b},
        )
        self.assertNotEqual(
            [(row["image_role"], row["image_filename"]) for row in mouse_a_1],
            [(row["image_role"], row["image_filename"]) for row in mouse_b],
        )

    def test_exact_90_percent_500_trial_plan(self):
        assignment, _, _, _ = protocol.create_or_load_assignment(
            "MOUSE_A", self.images, self.temp_dir
        )
        trials, _ = protocol.make_trial_plan(
            assignment,
            n_blocks=10,
            iti_min_sec=2.5,
            iti_max_sec=4.5,
            sequence_seed=12345,
            mouse_id="MOUSE_A",
        )
        self.assertEqual(len(trials), 500)
        self.assertTrue(
            all(
                trials[index]["image_filename"]
                != trials[index - 1]["image_filename"]
                for index in range(1, len(trials))
            )
        )
        for role in protocol.REWARDED_HIGH_ROLES:
            role_trials = [trial for trial in trials if trial["image_role"] == role]
            self.assertEqual(len(role_trials), 100)
            self.assertEqual(sum(t["reward_scheduled"] for t in role_trials), 90)
            self.assertEqual(sum(t["reward_omission_scheduled"] for t in role_trials), 10)
        for role in protocol.UNREWARDED_HIGH_ROLES:
            self.assertEqual(sum(t["image_role"] == role for t in trials), 100)
        for role in protocol.LOW_ROLES:
            self.assertEqual(sum(t["image_role"] == role for t in trials), 10)

    def test_fixed_90_percent_one_block_plan(self):
        assignment, _, _, _ = protocol.create_or_load_assignment(
            "MOUSE_A", self.images, self.temp_dir
        )
        trials, _ = protocol.make_trial_plan(
            assignment,
            n_blocks=1,
            iti_min_sec=1.0,
            iti_max_sec=1.1,
            sequence_seed=67890,
            mouse_id="MOUSE_A",
        )
        self.assertEqual(protocol.REWARD_PROBABILITY, 0.90)
        for role in protocol.REWARDED_HIGH_ROLES:
            role_trials = [trial for trial in trials if trial["image_role"] == role]
            self.assertEqual(len(role_trials), 10)
            self.assertEqual(sum(t["reward_scheduled"] for t in role_trials), 9)
            self.assertEqual(sum(t["reward_omission_scheduled"] for t in role_trials), 1)

    def test_no_reward_can_appear_on_ineligible_trial(self):
        assignment, _, _, _ = protocol.create_or_load_assignment(
            "MOUSE_A", self.images, self.temp_dir
        )
        trials, _ = protocol.make_trial_plan(
            assignment,
            n_blocks=2,
            iti_min_sec=1.0,
            iti_max_sec=1.1,
            sequence_seed=42,
            mouse_id="MOUSE_A",
        )
        self.assertTrue(
            all(
                trial["reward_eligible"] or not trial["reward_scheduled"]
                for trial in trials
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
