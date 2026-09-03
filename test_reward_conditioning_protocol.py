#!/usr/bin/env python3
"""Hardware-free checks for the exposure/reversal protocol."""

import json
import random
import shutil
import tempfile
import threading
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import reward_conditioning_protocol as protocol


class RewardConditioningProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="exposure_reward_test_"))
        self.images = [Path("natural_image_%04d.png" % index) for index in range(100)]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def assert_no_json_temps(self, directory=None):
        directory = self.temp_dir if directory is None else Path(directory)
        self.assertEqual(list(directory.glob(".*.tmp")), [])

    @staticmethod
    def assignment_mapping(rows):
        return [(row["image_role"], row["image_filename"]) for row in rows]

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

    def test_atomic_global_panel_creation_and_reuse(self):
        panel, path, created = protocol.create_or_load_global_panel(self.images, self.temp_dir)
        payload = json.loads(path.read_text())
        self.assertTrue(created)
        self.assertEqual(payload["schema_version"], protocol.PANEL_SCHEMA_VERSION)
        self.assertEqual(payload["protocol_version"], protocol.PROTOCOL_VERSION)
        self.assertEqual([item.name for item in panel], payload["image_filenames"])
        expected = sorted(random.Random(protocol.DEFAULT_GLOBAL_PANEL_SEED).sample(sorted(self.images), 14))
        self.assertEqual(panel, expected)
        with mock.patch.object(protocol, "atomic_write_json") as writer:
            reused, reused_path, reused_created = protocol.create_or_load_global_panel(list(reversed(self.images)), self.temp_dir, panel_seed=999)
        self.assertFalse(reused_created)
        self.assertEqual(reused_path, path)
        self.assertEqual(reused, panel)
        writer.assert_not_called()
        self.assert_no_json_temps()

    def test_global_panel_protocol_version_is_required(self):
        _, path, _ = protocol.create_or_load_global_panel(self.images, self.temp_dir)
        payload = json.loads(path.read_text())
        for version in (None, "wrong_protocol"):
            with self.subTest(version=version):
                bad = dict(payload)
                if version is None:
                    bad.pop("protocol_version", None)
                else:
                    bad["protocol_version"] = version
                protocol.atomic_write_json(path, bad)
                with self.assertRaisesRegex(RuntimeError, "protocol version mismatch"):
                    protocol.create_or_load_global_panel(self.images, self.temp_dir)

    def test_assignment_row_protocol_version_is_required(self):
        rows = self.assignment()
        protocol.validate_assignment_rows(rows)
        for version in (None, "wrong_protocol"):
            with self.subTest(version=version):
                bad = [dict(row) for row in rows]
                if version is None:
                    bad[0].pop("protocol_version", None)
                else:
                    bad[0]["protocol_version"] = version
                with self.assertRaisesRegex(RuntimeError, "Assignment row belongs"):
                    protocol.validate_assignment_rows(bad)

    def test_failed_atomic_writes_preserve_old_file(self):
        path = protocol.global_panel_path(self.temp_dir)
        old = b'{"authoritative": true}\n'
        path.write_bytes(old)
        with mock.patch.object(protocol.os, "replace", side_effect=OSError("replace")):
            with self.assertRaises(OSError):
                protocol.atomic_write_json(path, {"replacement": True})
        self.assertEqual(path.read_bytes(), old)
        self.assert_no_json_temps()
        with mock.patch.object(protocol.os, "fsync", side_effect=OSError("fsync")):
            with self.assertRaises(OSError):
                protocol.atomic_write_json(path, {"replacement": True})
        self.assertEqual(path.read_bytes(), old)
        self.assert_no_json_temps()

    def test_concurrent_panel_creation_converges(self):
        barrier = threading.Barrier(2)
        def create():
            barrier.wait(timeout=5)
            return protocol.create_or_load_global_panel(self.images, self.temp_dir)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: create(), range(2)))
        self.assertEqual([item.name for item in results[0][0]], [item.name for item in results[1][0]])
        self.assertEqual(sorted(result[2] for result in results), [False, True])

    def test_concurrent_assignment_creation_is_safe(self):
        barrier = threading.Barrier(2)
        def create(mouse):
            barrier.wait(timeout=5)
            return protocol.create_or_load_assignment(mouse, self.images, self.temp_dir)
        with ThreadPoolExecutor(max_workers=2) as pool:
            same = list(pool.map(lambda _: create("RACE"), range(2)))
        self.assertEqual(self.assignment_mapping(same[0][0]), self.assignment_mapping(same[1][0]))
        self.assertEqual(sorted(item[2] for item in same), [False, True])
        with ThreadPoolExecutor(max_workers=2) as pool:
            different = list(pool.map(create, ("A", "B")))
        self.assertTrue(all(item[2] for item in different))
        self.assertEqual({row["image_filename"] for row in different[0][0]}, {row["image_filename"] for row in different[1][0]})

    def test_existing_assignment_is_authoritative_under_changed_inputs(self):
        first = protocol.create_or_load_assignment("FIXED", self.images, self.temp_dir, master_seed=111, panel_seed=222)
        assignment_bytes = first[1].read_bytes()
        panel_bytes = protocol.global_panel_path(self.temp_dir).read_bytes()
        with mock.patch.object(protocol, "atomic_write_json") as writer:
            second = protocol.create_or_load_assignment("FIXED", list(reversed(self.images)), self.temp_dir, master_seed=999, panel_seed=888)
        self.assertEqual(self.assignment_mapping(first[0]), self.assignment_mapping(second[0]))
        self.assertEqual(first[3], second[3])
        self.assertEqual(first[1].read_bytes(), assignment_bytes)
        self.assertEqual(protocol.global_panel_path(self.temp_dir).read_bytes(), panel_bytes)
        writer.assert_not_called()

    def test_off_panel_image_is_rejected_without_repair(self):
        rows, path, _, _ = protocol.create_or_load_assignment("OFF", self.images, self.temp_dir)
        payload = json.loads(path.read_text())
        panel = json.loads(protocol.global_panel_path(self.temp_dir).read_text())
        replacement = next(item for item in self.images if item.name not in panel["image_filenames"])
        payload["images"][0]["image_filename"] = replacement.name
        payload["images"][0]["image_path"] = str(replacement)
        protocol.atomic_write_json(path, payload)
        bad = path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "do not match"):
            protocol.create_or_load_assignment("OFF", self.images, self.temp_dir)
        self.assertEqual(path.read_bytes(), bad)

    def test_panel_provenance_and_missing_panel_are_safe(self):
        _, path, _, _ = protocol.create_or_load_assignment("PROVENANCE", self.images, self.temp_dir)
        payload = json.loads(path.read_text())
        payload["global_panel_path"] = "/old/panel.json"
        protocol.atomic_write_json(path, payload)
        with self.assertRaisesRegex(RuntimeError, "inconsistent global panel path"):
            protocol.create_or_load_assignment("PROVENANCE", self.images, self.temp_dir)
        panel_path = protocol.global_panel_path(self.temp_dir)
        panel_path.unlink()
        with self.assertRaisesRegex(RuntimeError, "Refusing to create"):
            protocol.create_or_load_assignment("PROVENANCE", self.images, self.temp_dir)
        self.assertFalse(panel_path.exists())

    def test_corrupt_panel_and_assignment_fail_without_replacement(self):
        panel_path = protocol.global_panel_path(self.temp_dir)
        panel_path.write_text("not-json")
        with self.assertRaises(RuntimeError):
            protocol.create_or_load_global_panel(self.images, self.temp_dir)
        self.assertEqual(panel_path.read_text(), "not-json")
        panel_path.unlink()
        rows, assignment_path, _, _ = protocol.create_or_load_assignment("BAD", self.images, self.temp_dir)
        assignment_path.write_text("not-json")
        with self.assertRaises(RuntimeError):
            protocol.create_or_load_assignment("BAD", self.images, self.temp_dir)
        self.assertEqual(assignment_path.read_text(), "not-json")

    def test_missing_saved_image_is_rejected(self):
        rows, path, _, _ = protocol.create_or_load_assignment("MISSING", self.images, self.temp_dir)
        bad_bytes = path.read_bytes()
        available = [item for item in self.images if item.name != rows[0]["image_filename"]]
        with self.assertRaisesRegex(RuntimeError, "missing image"):
            protocol.create_or_load_assignment("MISSING", available, self.temp_dir)
        self.assertEqual(path.read_bytes(), bad_bytes)

    def test_force_new_is_explicit_and_lock_releases(self):
        rows, path, _, seed = protocol.create_or_load_assignment("FORCE", self.images, self.temp_dir, master_seed=1)
        old_bytes = path.read_bytes()
        with mock.patch.object(protocol.os, "replace", side_effect=OSError("replace")):
            with self.assertRaises(OSError):
                protocol.create_or_load_assignment("FORCE", self.images, self.temp_dir, master_seed=2, force_new=True)
        self.assertEqual(path.read_bytes(), old_bytes)
        loaded = protocol.create_or_load_assignment("FORCE", self.images, self.temp_dir)
        self.assertEqual(self.assignment_mapping(loaded[0]), self.assignment_mapping(rows))
        self.assertEqual(loaded[3], seed)
        with self.assertRaisesRegex(RuntimeError, "lock failure"):
            with protocol.assignment_directory_lock(self.temp_dir):
                raise RuntimeError("lock failure")
        with protocol.assignment_directory_lock(self.temp_dir):
            pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
