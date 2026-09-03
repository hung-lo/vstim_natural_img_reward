#!/usr/bin/env python3
"""Hardware-free checks for the exposure/reversal protocol."""

import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import reward_conditioning_protocol as protocol


class RewardConditioningProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="exposure_reward_test_"))
        self.images = [Path("natural_image_%04d.png" % index) for index in range(100)]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def assignment(self, mouse="MOUSE"):
        return protocol.create_or_load_assignment(mouse, self.images, self.temp_dir)[0]

    def plan(self, phase="acquisition", seed=12345):
        return protocol.make_trial_plan(
            self.assignment(), 10, sequence_seed=seed,
            mouse_id="MOUSE", contingency_phase=phase,
        )[0]

    def test_role_structure_and_namespaced_assignment(self):
        rows, path, created, _ = protocol.create_or_load_assignment(
            "MOUSE", self.images, self.temp_dir,
        )
        self.assertTrue(created)
        self.assertEqual(path.name, "MOUSE_exposure_reward_v1_assignment.json")
        self.assertEqual(len(protocol.ALL_ROLES), 14)
        self.assertEqual([r["exposure_level"] for r in rows].count("high"), 4)
        self.assertEqual([r["exposure_level"] for r in rows].count("medium"), 4)
        self.assertEqual([r["exposure_level"] for r in rows].count("low"), 6)
        self.assertNotIn("reward_eligible", rows[0])
        self.assertEqual(len({r["image_filename"] for r in rows}), 14)
        loaded, _, created_again, _ = protocol.create_or_load_assignment("MOUSE", list(reversed(self.images)), self.temp_dir)
        self.assertFalse(created_again)
        self.assertEqual([(r["image_role"], r["image_filename"]) for r in rows], [(r["image_role"], r["image_filename"]) for r in loaded])

    def test_different_mice_are_counterbalanced_over_shared_panel(self):
        a = self.assignment("A")
        b = self.assignment("B")
        self.assertEqual({r["image_filename"] for r in a}, {r["image_filename"] for r in b})
        self.assertNotEqual([(r["image_role"], r["image_filename"]) for r in a], [(r["image_role"], r["image_filename"]) for r in b])

    def test_exact_block_and_session_exposure_counts(self):
        trials = self.plan()
        self.assertEqual(len(trials), 500)
        self.assertEqual(sum(t["image_role"] == protocol.HIGH_ROLES[0] for t in trials), 80)
        self.assertEqual(sum(t["image_role"] == protocol.MEDIUM_ROLES[0] for t in trials), 30)
        self.assertEqual(sum(t["image_role"] == protocol.LOW_ROLES[0] for t in trials), 10)
        for block in range(10):
            rows = [t for t in trials if t["block_index"] == block]
            self.assertEqual(len(rows), 50)
            self.assertEqual({r: sum(t["image_role"] == r for t in rows) for r in protocol.HIGH_ROLES}, {r: 8 for r in protocol.HIGH_ROLES})
            self.assertEqual({r: sum(t["image_role"] == r for t in rows) for r in protocol.MEDIUM_ROLES}, {r: 3 for r in protocol.MEDIUM_ROLES})
            self.assertEqual({r: sum(t["image_role"] == r for t in rows) for r in protocol.LOW_ROLES}, {r: 1 for r in protocol.LOW_ROLES})
        self.assertTrue(all(a["image_filename"] != b["image_filename"] for a, b in zip(trials, trials[1:])))

    def test_acquisition_and_reversal_reward_mapping(self):
        for phase in protocol.CONTINGENCY_PHASES:
            trials = self.plan(phase)
            self.assertEqual({t["image_role"] for t in trials if t["reward_eligible"]}, set(protocol.REWARDED_ROLES_BY_PHASE[phase]))
            self.assertEqual(sum(t["reward_scheduled"] for t in trials), 198)
            self.assertEqual(sum(t["reward_omission_scheduled"] for t in trials), 22)
            for role in protocol.REWARDED_ROLES_BY_PHASE[phase]:
                role_trials = [t for t in trials if t["image_role"] == role]
                self.assertEqual(sum(t["reward_scheduled"] for t in role_trials), 72 if role.startswith("high_") else 27)
                self.assertEqual(sum(t["reward_omission_scheduled"] for t in role_trials), 8 if role.startswith("high_") else 3)

    def test_reversal_changes_contingency_only(self):
        acquisition = self.plan("acquisition", seed=77)
        reversal = self.plan("reversal", seed=77)
        permanent = ("image_role", "exposure_level", "reward_trajectory", "presentation_probability")
        self.assertEqual(Counter(tuple(t[field] for field in permanent) for t in acquisition), Counter(tuple(t[field] for field in permanent) for t in reversal))
        self.assertEqual([t["image_filename"] for t in acquisition], [t["image_filename"] for t in reversal])
        self.assertNotEqual([t["reward_eligible"] for t in acquisition], [t["reward_eligible"] for t in reversal])

    def test_legacy_files_are_not_reused(self):
        (self.temp_dir / "MOUSE_reward_conditioning_assignment.json").write_text(json.dumps({"schema_version": 2}))
        rows, path, _, _ = protocol.create_or_load_assignment("MOUSE", self.images, self.temp_dir)
        self.assertEqual(path.name, "MOUSE_exposure_reward_v1_assignment.json")
        self.assertEqual(len(rows), 14)

    def test_invalid_non_default_block_multiple_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "multiple of 10"):
            protocol.make_trial_plan(self.assignment(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
