#!/usr/bin/env python3
"""Hardware-free regression tests for the reward-conditioning planner."""

import json
import random
import shutil
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import reward_conditioning_protocol as protocol


class RewardConditioningProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="reward_conditioning_test_"))
        self.images = [Path("natural_image_%04d.png" % index) for index in range(100)]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def assert_no_json_temps(self, directory=None):
        directory = self.temp_dir if directory is None else Path(directory)
        self.assertEqual(list(directory.glob(".*.tmp")), [])

    @staticmethod
    def assignment_mapping(rows):
        return [(row["image_role"], row["image_filename"]) for row in rows]

    def test_atomic_global_panel_creation_and_reuse(self):
        panel_1, panel_path, created_1 = protocol.create_or_load_global_panel(
            self.images, self.temp_dir
        )
        first_bytes = panel_path.read_bytes()
        payload = json.loads(first_bytes.decode("utf-8"))

        self.assertTrue(created_1)
        self.assertEqual(payload["schema_version"], protocol.PANEL_SCHEMA_VERSION)
        self.assertEqual(len(payload["image_filenames"]), 14)
        self.assertEqual(len(set(payload["image_filenames"])), 14)
        self.assertEqual(
            [path.name for path in panel_1], payload["image_filenames"]
        )
        expected_panel = sorted(
            random.Random(protocol.DEFAULT_GLOBAL_PANEL_SEED).sample(
                sorted(self.images), len(protocol.ALL_ROLES)
            )
        )
        self.assertEqual(panel_1, expected_panel)
        self.assert_no_json_temps()

        with mock.patch.object(protocol, "atomic_write_json") as atomic_write:
            panel_2, second_path, created_2 = protocol.create_or_load_global_panel(
                list(reversed(self.images)), self.temp_dir, panel_seed=123456
            )

        self.assertFalse(created_2)
        self.assertEqual(second_path, panel_path)
        self.assertEqual(panel_2, panel_1)
        self.assertEqual(panel_path.read_bytes(), first_bytes)
        atomic_write.assert_not_called()
        self.assert_no_json_temps()

    def test_atomic_write_failures_preserve_old_file_and_clean_temps(self):
        panel_path = protocol.global_panel_path(self.temp_dir)
        old_bytes = b'{"authoritative": true}\n'
        panel_path.write_bytes(old_bytes)

        with mock.patch.object(
            protocol.os, "replace", side_effect=OSError("synthetic replace failure")
        ):
            with self.assertRaises(OSError):
                protocol.atomic_write_json(panel_path, {"replacement": True})
        self.assertEqual(panel_path.read_bytes(), old_bytes)
        self.assert_no_json_temps()

        with mock.patch.object(
            protocol.os, "fsync", side_effect=OSError("synthetic fsync failure")
        ):
            with self.assertRaises(OSError):
                protocol.atomic_write_json(panel_path, {"replacement": True})
        self.assertEqual(panel_path.read_bytes(), old_bytes)
        self.assert_no_json_temps()

    def test_concurrent_global_panel_creation_converges(self):
        barrier = threading.Barrier(2)
        original_atomic_write = protocol.atomic_write_json

        def delayed_atomic_write(path, payload):
            time.sleep(0.05)
            return original_atomic_write(path, payload)

        def create_panel():
            barrier.wait(timeout=5.0)
            return protocol.create_or_load_global_panel(self.images, self.temp_dir)

        with mock.patch.object(
            protocol, "atomic_write_json", side_effect=delayed_atomic_write
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: create_panel(), range(2)))

        panels = [[path.name for path in result[0]] for result in results]
        self.assertEqual(panels[0], panels[1])
        self.assertEqual(sorted(result[2] for result in results), [False, True])
        payload = json.loads(protocol.global_panel_path(self.temp_dir).read_text())
        self.assertEqual(payload["image_filenames"], panels[0])
        self.assert_no_json_temps()

    def test_concurrent_same_mouse_assignment_converges(self):
        protocol.create_or_load_global_panel(self.images, self.temp_dir)
        barrier = threading.Barrier(2)
        original_atomic_write = protocol.atomic_write_json

        def delayed_atomic_write(path, payload):
            time.sleep(0.05)
            return original_atomic_write(path, payload)

        def create_assignment():
            barrier.wait(timeout=5.0)
            return protocol.create_or_load_assignment(
                "MOUSE_RACE", self.images, self.temp_dir
            )

        with mock.patch.object(
            protocol, "atomic_write_json", side_effect=delayed_atomic_write
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(
                    executor.map(lambda _: create_assignment(), range(2))
                )

        mappings = [self.assignment_mapping(result[0]) for result in results]
        self.assertEqual(mappings[0], mappings[1])
        self.assertEqual(sorted(result[2] for result in results), [False, True])
        saved_path = protocol.assignment_path_for_mouse(
            self.temp_dir, "MOUSE_RACE"
        )
        self.assertEqual(
            self.assignment_mapping(json.loads(saved_path.read_text())["images"]),
            mappings[0],
        )
        self.assert_no_json_temps()

    def test_concurrent_different_mouse_assignments_are_safe(self):
        panel, _, _ = protocol.create_or_load_global_panel(
            self.images, self.temp_dir
        )
        barrier = threading.Barrier(2)

        def create_assignment(mouse_id):
            barrier.wait(timeout=5.0)
            return protocol.create_or_load_assignment(
                mouse_id, self.images, self.temp_dir
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(create_assignment, mouse_id)
                for mouse_id in ("MOUSE_A", "MOUSE_B")
            ]
            results = [future.result() for future in futures]

        panel_names = {path.name for path in panel}
        for mouse_id, result in zip(("MOUSE_A", "MOUSE_B"), results):
            rows, assignment_path, created, _ = result
            self.assertTrue(created)
            self.assertEqual(
                assignment_path,
                protocol.assignment_path_for_mouse(self.temp_dir, mouse_id),
            )
            self.assertEqual({row["image_filename"] for row in rows}, panel_names)
            json.loads(assignment_path.read_text())
        self.assert_no_json_temps()

    def test_existing_assignment_is_authoritative_under_changed_inputs(self):
        rows_1, assignment_path, created_1, seed_1 = (
            protocol.create_or_load_assignment(
                "MOUSE_FIXED",
                self.images,
                self.temp_dir,
                master_seed=111,
                panel_seed=222,
            )
        )
        first_bytes = assignment_path.read_bytes()
        panel_path = protocol.global_panel_path(self.temp_dir)
        panel_bytes = panel_path.read_bytes()

        with mock.patch.object(protocol, "atomic_write_json") as atomic_write:
            rows_2, _, created_2, seed_2 = protocol.create_or_load_assignment(
                "MOUSE_FIXED",
                list(reversed(self.images)),
                self.temp_dir,
                master_seed=999,
                panel_seed=888,
            )

        self.assertTrue(created_1)
        self.assertFalse(created_2)
        self.assertEqual(self.assignment_mapping(rows_1), self.assignment_mapping(rows_2))
        self.assertEqual(seed_1, seed_2)
        self.assertEqual(assignment_path.read_bytes(), first_bytes)
        self.assertEqual(panel_path.read_bytes(), panel_bytes)
        atomic_write.assert_not_called()

    def test_saved_assignment_with_off_panel_image_is_rejected(self):
        _, assignment_path, _, _ = protocol.create_or_load_assignment(
            "MOUSE_OFF_PANEL", self.images, self.temp_dir
        )
        panel_path = protocol.global_panel_path(self.temp_dir)
        panel_payload = json.loads(panel_path.read_text(encoding="utf-8"))
        assignment_payload = json.loads(
            assignment_path.read_text(encoding="utf-8")
        )
        replacement = next(
            path
            for path in self.images
            if path.name not in set(panel_payload["image_filenames"])
        )
        assignment_payload["images"][0]["image_filename"] = replacement.name
        assignment_payload["images"][0]["image_path"] = str(replacement)
        protocol.atomic_write_json(assignment_path, assignment_payload)
        altered_assignment_bytes = assignment_path.read_bytes()
        panel_bytes = panel_path.read_bytes()

        with self.assertRaisesRegex(RuntimeError, "do not match"):
            protocol.create_or_load_assignment(
                "MOUSE_OFF_PANEL", self.images, self.temp_dir
            )

        self.assertEqual(assignment_path.read_bytes(), altered_assignment_bytes)
        self.assertEqual(panel_path.read_bytes(), panel_bytes)

    def test_assignment_panel_provenance_is_checked_migration_safely(self):
        _, assignment_path, _, _ = protocol.create_or_load_assignment(
            "MOUSE_PROVENANCE", self.images, self.temp_dir
        )
        payload = json.loads(assignment_path.read_text(encoding="utf-8"))
        authoritative_seed = payload["global_panel_seed"]
        payload["global_panel_path"] = str(
            Path("/previous/assignment/location") / protocol.GLOBAL_PANEL_FILENAME
        )
        protocol.atomic_write_json(assignment_path, payload)

        with mock.patch.object(protocol, "atomic_write_json") as atomic_write:
            _, _, created, _ = protocol.create_or_load_assignment(
                "MOUSE_PROVENANCE", self.images, self.temp_dir
            )
        self.assertFalse(created)
        atomic_write.assert_not_called()

        payload["global_panel_seed"] = authoritative_seed + 1
        protocol.atomic_write_json(assignment_path, payload)
        mismatched_seed_bytes = assignment_path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "seed"):
            protocol.create_or_load_assignment(
                "MOUSE_PROVENANCE", self.images, self.temp_dir
            )
        self.assertEqual(assignment_path.read_bytes(), mismatched_seed_bytes)

        payload["global_panel_seed"] = authoritative_seed
        payload["global_panel_path"] = "/previous/location/wrong_panel.json"
        protocol.atomic_write_json(assignment_path, payload)
        mismatched_path_bytes = assignment_path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "panel path"):
            protocol.create_or_load_assignment(
                "MOUSE_PROVENANCE", self.images, self.temp_dir
            )
        self.assertEqual(assignment_path.read_bytes(), mismatched_path_bytes)

    def test_existing_assignment_with_missing_panel_fails_without_repair(self):
        _, assignment_path, _, _ = protocol.create_or_load_assignment(
            "MOUSE_ORPHAN", self.images, self.temp_dir
        )
        assignment_bytes = assignment_path.read_bytes()
        panel_path = protocol.global_panel_path(self.temp_dir)
        panel_path.unlink()

        with self.assertRaisesRegex(RuntimeError, "global panel is missing"):
            protocol.create_or_load_assignment(
                "MOUSE_ORPHAN", self.images, self.temp_dir
            )

        self.assertFalse(panel_path.exists())
        self.assertEqual(assignment_path.read_bytes(), assignment_bytes)

    def test_orphaned_assignment_blocks_new_mouse_and_direct_panel_creation(self):
        _, assignment_path, _, _ = protocol.create_or_load_assignment(
            "MOUSE_A", self.images, self.temp_dir
        )
        assignment_bytes = assignment_path.read_bytes()
        panel_path = protocol.global_panel_path(self.temp_dir)
        panel_path.unlink()

        with self.assertRaisesRegex(RuntimeError, "Refusing to regenerate"):
            protocol.create_or_load_global_panel(self.images, self.temp_dir)
        with self.assertRaisesRegex(RuntimeError, "Refusing to regenerate"):
            protocol.create_or_load_assignment(
                "MOUSE_B", self.images, self.temp_dir
            )

        self.assertFalse(panel_path.exists())
        self.assertFalse(
            protocol.assignment_path_for_mouse(self.temp_dir, "MOUSE_B").exists()
        )
        self.assertEqual(assignment_path.read_bytes(), assignment_bytes)

    def test_corrupt_or_incompatible_panel_fails_without_replacement(self):
        for case_name, bad_bytes in (
            ("corrupt", b"not-json\n"),
            ("wrong_schema", b'{"schema_version": 999, "image_filenames": []}\n'),
        ):
            case_dir = self.temp_dir / case_name
            case_dir.mkdir()
            panel_path = protocol.global_panel_path(case_dir)
            panel_path.write_bytes(bad_bytes)
            with self.subTest(case=case_name):
                with self.assertRaises(RuntimeError):
                    protocol.create_or_load_global_panel(self.images, case_dir)
                self.assertEqual(panel_path.read_bytes(), bad_bytes)

    def test_global_panel_requires_integer_seed_metadata(self):
        for case_name, bad_seed in (
            ("missing_seed", None),
            ("string_seed", "not-an-integer"),
            ("float_seed", 7.5),
            ("boolean_seed", True),
        ):
            case_dir = self.temp_dir / case_name
            _, panel_path, _ = protocol.create_or_load_global_panel(
                self.images, case_dir
            )
            payload = json.loads(panel_path.read_text(encoding="utf-8"))
            if bad_seed is None:
                del payload["panel_seed"]
            else:
                payload["panel_seed"] = bad_seed
            protocol.atomic_write_json(panel_path, payload)
            bad_bytes = panel_path.read_bytes()

            with self.subTest(case=case_name):
                with self.assertRaises(RuntimeError):
                    protocol.create_or_load_global_panel(self.images, case_dir)
                self.assertEqual(panel_path.read_bytes(), bad_bytes)

    def test_corrupt_or_incompatible_assignment_fails_without_regeneration(self):
        bad_payloads = (
            b"not-json\n",
            b'{"schema_version": 999, "mouse_id": "MOUSE_BAD", "images": []}\n',
            (
                b'{"schema_version": 2, "mouse_id": "SOMEONE_ELSE", '
                b'"images": []}\n'
            ),
        )
        for index, bad_bytes in enumerate(bad_payloads):
            case_dir = self.temp_dir / ("bad_assignment_%d" % index)
            protocol.create_or_load_assignment("MOUSE_BAD", self.images, case_dir)
            assignment_path = protocol.assignment_path_for_mouse(
                case_dir, "MOUSE_BAD"
            )
            assignment_path.write_bytes(bad_bytes)
            with self.subTest(index=index):
                with self.assertRaises(RuntimeError):
                    protocol.create_or_load_assignment(
                        "MOUSE_BAD", self.images, case_dir
                    )
                self.assertEqual(assignment_path.read_bytes(), bad_bytes)

    def test_missing_saved_images_fail_without_replacement(self):
        panel, panel_path, _ = protocol.create_or_load_global_panel(
            self.images, self.temp_dir
        )
        panel_bytes = panel_path.read_bytes()
        available_without_panel_image = [
            path for path in self.images if path.name != panel[0].name
        ]
        with self.assertRaises(RuntimeError):
            protocol.create_or_load_global_panel(
                available_without_panel_image, self.temp_dir
            )
        self.assertEqual(panel_path.read_bytes(), panel_bytes)

        rows, assignment_path, _, _ = protocol.create_or_load_assignment(
            "MOUSE_MISSING", self.images, self.temp_dir
        )
        assignment_bytes = assignment_path.read_bytes()
        available_without_assignment_image = [
            path for path in self.images if path.name != rows[0]["image_filename"]
        ]
        with self.assertRaises(RuntimeError):
            protocol.create_or_load_assignment(
                "MOUSE_MISSING", available_without_assignment_image, self.temp_dir
            )
        self.assertEqual(assignment_path.read_bytes(), assignment_bytes)

    def test_force_new_is_only_assignment_replacement_path(self):
        _, assignment_path, _, _ = protocol.create_or_load_assignment(
            "MOUSE_FORCE", self.images, self.temp_dir, master_seed=111
        )
        original_bytes = assignment_path.read_bytes()
        panel_path = protocol.global_panel_path(self.temp_dir)
        panel_bytes = panel_path.read_bytes()
        panel_payload = json.loads(panel_bytes.decode("utf-8"))
        original_atomic_write = protocol.atomic_write_json

        with mock.patch.object(
            protocol, "atomic_write_json", wraps=original_atomic_write
        ) as atomic_write:
            _, _, created, _ = protocol.create_or_load_assignment(
                "MOUSE_FORCE", self.images, self.temp_dir, master_seed=222
            )
            self.assertFalse(created)
            atomic_write.assert_not_called()

            _, _, created, _ = protocol.create_or_load_assignment(
                "MOUSE_FORCE",
                self.images,
                self.temp_dir,
                master_seed=222,
                panel_seed=987654,
                force_new=True,
            )
            self.assertTrue(created)
            self.assertEqual(atomic_write.call_count, 1)
            self.assertEqual(Path(atomic_write.call_args[0][0]), assignment_path)

        self.assertNotEqual(assignment_path.read_bytes(), original_bytes)
        self.assertEqual(panel_path.read_bytes(), panel_bytes)
        replacement_payload = json.loads(
            assignment_path.read_text(encoding="utf-8")
        )
        self.assertEqual(
            {row["image_filename"] for row in replacement_payload["images"]},
            set(panel_payload["image_filenames"]),
        )
        self.assertEqual(
            replacement_payload["global_panel_seed"], panel_payload["panel_seed"]
        )

    def test_failed_force_new_preserves_previous_assignment(self):
        rows, assignment_path, _, old_seed = protocol.create_or_load_assignment(
            "MOUSE_FORCE_FAIL", self.images, self.temp_dir, master_seed=111
        )
        old_bytes = assignment_path.read_bytes()

        with mock.patch.object(
            protocol.os, "replace", side_effect=OSError("synthetic replace failure")
        ):
            with self.assertRaises(OSError):
                protocol.create_or_load_assignment(
                    "MOUSE_FORCE_FAIL",
                    self.images,
                    self.temp_dir,
                    master_seed=222,
                    force_new=True,
                )

        self.assertEqual(assignment_path.read_bytes(), old_bytes)
        loaded_rows, _, created, loaded_seed = protocol.create_or_load_assignment(
            "MOUSE_FORCE_FAIL", self.images, self.temp_dir
        )
        self.assertFalse(created)
        self.assertEqual(self.assignment_mapping(loaded_rows), self.assignment_mapping(rows))
        self.assertEqual(loaded_seed, old_seed)
        self.assert_no_json_temps()

    def test_missing_or_corrupt_panel_blocks_force_new_assignment(self):
        for case_name in ("missing", "corrupt"):
            case_dir = self.temp_dir / ("force_new_" + case_name)
            _, assignment_path, _, _ = protocol.create_or_load_assignment(
                "MOUSE_FORCE_BLOCKED", self.images, case_dir
            )
            assignment_bytes = assignment_path.read_bytes()
            panel_path = protocol.global_panel_path(case_dir)
            if case_name == "missing":
                panel_path.unlink()
                expected_panel_bytes = None
            else:
                panel_path.write_bytes(b"not-json\n")
                expected_panel_bytes = panel_path.read_bytes()

            with self.subTest(case=case_name):
                with mock.patch.object(protocol, "atomic_write_json") as atomic_write:
                    with self.assertRaises(RuntimeError):
                        protocol.create_or_load_assignment(
                            "MOUSE_FORCE_BLOCKED",
                            self.images,
                            case_dir,
                            master_seed=999,
                            force_new=True,
                        )
                    atomic_write.assert_not_called()
                self.assertEqual(assignment_path.read_bytes(), assignment_bytes)
                if expected_panel_bytes is None:
                    self.assertFalse(panel_path.exists())
                else:
                    self.assertEqual(panel_path.read_bytes(), expected_panel_bytes)

    def test_assignment_directory_lock_releases_after_exception(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic lock failure"):
            with protocol.assignment_directory_lock(self.temp_dir):
                raise RuntimeError("synthetic lock failure")

        with protocol.assignment_directory_lock(self.temp_dir) as lock_path:
            self.assertEqual(
                lock_path,
                self.temp_dir / protocol.ASSIGNMENT_LOCK_FILENAME,
            )

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
        assignment_payload = json.loads(path_a.read_text(encoding="utf-8"))
        panel_payload = json.loads(
            protocol.global_panel_path(self.temp_dir).read_text(encoding="utf-8")
        )
        self.assertEqual(
            assignment_payload["schema_version"],
            protocol.ASSIGNMENT_SCHEMA_VERSION,
        )
        self.assertEqual(
            {row["image_role"] for row in assignment_payload["images"]},
            set(protocol.ALL_ROLES),
        )
        self.assertEqual(
            {row["image_filename"] for row in assignment_payload["images"]},
            set(panel_payload["image_filenames"]),
        )
        expected_role_order = [Path(name) for name in panel_payload["image_filenames"]]
        random.Random(
            protocol.stable_seed(
                "reward-conditioning-role-assignment",
                protocol.DEFAULT_ASSIGNMENT_MASTER_SEED,
                "MOUSE_A",
            )
        ).shuffle(expected_role_order)
        self.assertEqual(
            [row["image_filename"] for row in mouse_a_1],
            [path.name for path in expected_role_order],
        )
        self.assert_no_json_temps()

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
        conditioned_trials = [trial for trial in trials if trial["reward_eligible"]]
        self.assertEqual(len(conditioned_trials), 20)
        self.assertEqual(sum(trial["suction_scheduled"] for trial in conditioned_trials), 20)
        self.assertEqual(sum(trial["reward_scheduled"] for trial in conditioned_trials), 18)
        self.assertEqual(sum(trial["reward_omission_scheduled"] for trial in conditioned_trials), 2)
        self.assertTrue(all(not trial["suction_scheduled"] for trial in trials if not trial["reward_eligible"]))
        self.assertTrue(all(trial["planned_suction_delay_sec"] == 3.5 for trial in trials))

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
