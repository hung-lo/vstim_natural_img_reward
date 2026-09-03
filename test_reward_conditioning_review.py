#!/usr/bin/env python3
"""Regression tests for persistent state, recent mice, and behavior readouts."""

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import PIL  # noqa: F401
except ImportError:
    import sys
    import types
    pil_stub = types.ModuleType("PIL")
    pil_stub.Image = pil_stub.ImageDraw = pil_stub.ImageOps = object
    sys.modules["PIL"] = pil_stub

import run_stringer_reward_conditioning as runner


class ReviewFixTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="exposure_reward_review_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def args(self, **overrides):
        values = {
            "simulate_gpio": False,
            "simulate_water_test": False,
            "simulate_behavior_test": False,
            "blocks": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_real_completion_updates_acquisition_then_first_reversal_once(self):
        with mock.patch.object(runner, "ASSIGNMENT_DIR", self.temp_dir):
            runner.ensure_initial_mouse_state("M")
            runner.record_completed_session("M", "acquisition", "S1", "2026-09-02T20:00:00Z")
            state = runner.load_mouse_state("M", required=True)
            self.assertEqual(state["current_phase"], "acquisition")
            self.assertEqual(state["completed_session_count"], 1)
            runner.record_completed_session("M", "reversal", "S2", "2026-09-03T20:00:00Z")
            state = runner.load_mouse_state("M", required=True)
            self.assertEqual(state["current_phase"], "reversal")
            self.assertTrue(state["reversal_has_started"])
            self.assertEqual(state["first_reversal_session_id"], "S2")
            self.assertEqual(state["completed_session_count"], 2)
            runner.record_completed_session("M", "reversal", "S3", "2026-09-04T20:00:00Z")
            state = runner.load_mouse_state("M", required=True)
            self.assertEqual(state["first_reversal_session_id"], "S2")
            self.assertEqual(state["completed_session_count"], 3)

    def test_simulation_and_interruption_never_commit_state(self):
        real = self.args()
        simulated = self.args(simulate_gpio=True)
        behavior_simulated = self.args(simulate_behavior_test=True)
        self.assertTrue(runner.should_commit_persistent_mouse_state(real, {"simulate_gpio": False}, True))
        self.assertFalse(runner.should_commit_persistent_mouse_state(
            real, {"simulate_gpio": False}, True, protocol_qc_pass=False
        ))
        self.assertFalse(runner.should_commit_persistent_mouse_state(real, {"simulate_gpio": False}, False))
        self.assertFalse(runner.should_commit_persistent_mouse_state(simulated, {"simulate_gpio": True}, True))
        self.assertFalse(runner.should_commit_persistent_mouse_state(behavior_simulated, {"simulate_gpio": False}, True))
        with mock.patch.object(runner, "ASSIGNMENT_DIR", self.temp_dir):
            runner.ensure_initial_mouse_state("M")
            before = runner.state_path_for_mouse("M").read_bytes()
            self.assertFalse(runner.should_commit_persistent_mouse_state(simulated, {"simulate_gpio": True}, True))
            self.assertEqual(runner.state_path_for_mouse("M").read_bytes(), before)

    def test_assignment_and_state_mismatch_is_refused(self):
        with mock.patch.object(runner, "ASSIGNMENT_DIR", self.temp_dir):
            assignment = runner.assignment_path_for_mouse(self.temp_dir, "M")
            state = runner.state_path_for_mouse("M")
            assignment.write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "state is missing"):
                runner.mouse_protocol_files("M")
            assignment.unlink()
            state.write_text(json.dumps(runner.initial_mouse_state("M")))
            with self.assertRaisesRegex(RuntimeError, "assignment is missing"):
                runner.mouse_protocol_files("M")
            state.unlink()
            _, _, exists = runner.mouse_protocol_files("M")
            self.assertFalse(exists)
            assignment.write_text("{}")
            runner.ensure_initial_mouse_state("M")
            _, _, exists = runner.mouse_protocol_files("M")
            self.assertTrue(exists)

    def test_recent_mice_are_completed_only_newest_first_and_protocol_filtered(self):
        with mock.patch.object(runner, "ASSIGNMENT_DIR", self.temp_dir):
            runner.atomic_write_json(self.temp_dir / "new_state.json", runner.initial_mouse_state("NEW"))
            for mouse, stamp, phase in (
                ("OLD", "2026-09-01T20:00:00Z", "acquisition"),
                ("NEWEST", "2026-09-03T20:00:00Z", "reversal"),
                ("MIDDLE", "2026-09-02T20:00:00Z", "acquisition"),
            ):
                state = runner.initial_mouse_state(mouse)
                state.update(last_completed_utc_iso=stamp, current_phase=phase, completed_session_count=2)
                runner.atomic_write_json(runner.state_path_for_mouse(mouse), state)
            other = runner.initial_mouse_state("OTHER")
            other["protocol_version"] = "old"
            other["last_completed_utc_iso"] = "2026-09-04T20:00:00Z"
            runner.atomic_write_json(self.temp_dir / "OTHER_reward_conditioning_state.json", other)
            (self.temp_dir / "bad_reward_conditioning_state.json").write_text("not-json")
            warning = io.StringIO()
            with redirect_stdout(warning):
                states = runner.recent_mouse_states()
            self.assertEqual([state["mouse_id"] for state in states], ["NEWEST", "MIDDLE", "OLD"])
            self.assertEqual(warning.getvalue().count("WARNING:"), 1)
            self.assertIn("skipped 2 unreadable/incompatible", warning.getvalue())
            output = io.StringIO()
            with redirect_stdout(output):
                runner.print_recent_mice(limit=2)
            text = output.getvalue()
            self.assertIn("NEWEST", text)
            self.assertNotIn("OLD", text)
            self.assertNotIn("OTHER", text)

    def test_rollback_requires_override_and_confirmation(self):
        state = runner.initial_mouse_state("M")
        state["current_phase"] = "reversal"
        with self.assertRaisesRegex(RuntimeError, "rollback"):
            runner.resolve_contingency_phase(self.args(contingency_phase="acquisition"), state)
        with mock.patch.object(runner, "prompt_yes_no_strict", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                runner.resolve_contingency_phase(self.args(contingency_phase="acquisition", allow_phase_rollback=True), state)
        with mock.patch.object(runner, "prompt_yes_no_strict", return_value=True):
            self.assertEqual(runner.resolve_contingency_phase(self.args(contingency_phase="acquisition", allow_phase_rollback=True), state), "acquisition")

    def test_new_mouse_is_acquisition_only_and_blocks_nonstandard_real_lengths(self):
        self.assertEqual(runner.resolve_contingency_phase(self.args(), None), "acquisition")
        with self.assertRaisesRegex(RuntimeError, "new protocol mouse"):
            runner.resolve_contingency_phase(self.args(contingency_phase="reversal"), None)
        self.assertEqual(
            runner.resolve_session_block_count(self.args()), runner.DEFAULT_N_BLOCKS
        )
        with self.assertRaisesRegex(RuntimeError, "exactly 10 blocks / 500 trials"):
            runner.resolve_session_block_count(self.args(blocks=20))
        self.assertEqual(
            runner.resolve_session_block_count(self.args(simulate_gpio=True, blocks=20)), 20
        )

    @staticmethod
    def behavior_row(role, reward_eligible, full, late, completed=True, omission=False):
        return {
            "image_role": role,
            "exposure_level": "high" if role.startswith("high_") else "medium" if role.startswith("medium_") else "low",
            "reward_eligible": reward_eligible,
            "reward_omission_scheduled": omission,
            "trial_completed": completed,
            "stim_request_monotonic_ns": 1,
            "lick_count_0_to_1p0_sec": full,
            "lick_count_0p5_to_1p0_sec": late,
        }

    def test_partial_behavior_is_not_readiness_assessed(self):
        rows = [self.behavior_row("high_R_to_R", True, 1, 1)] * 8
        rows.append(self.behavior_row("high_R_to_R", True, 1, 1, completed=False))
        summary = runner.build_behavior_summary(rows, "acquisition")
        self.assertTrue(summary["partial"])
        self.assertIsNone(summary["reversal_readiness_candidate"])
        output = io.StringIO()
        with redirect_stdout(output):
            runner.print_behavior_summary(summary)
        self.assertIn("NOT ASSESSED (partial session)", output.getvalue())

    def test_behavior_thresholds_warnings_and_omission_denominator(self):
        rows = []
        for index in range(10):
            rows.append(self.behavior_row("high_R_to_R", True, index < 10, index < 5))
            rows.append(self.behavior_row("high_U_to_U", False, index < 5, index < 3))
        rows.extend([
            self.behavior_row("medium_R_to_R", True, 1, 1, omission=True),
            self.behavior_row("medium_R_to_R", True, 0, 0, omission=True),
            self.behavior_row("medium_R_to_R", True, 0, 0, omission=False),
        ])
        summary = runner.build_behavior_summary(rows, "acquisition")
        self.assertTrue(summary["reversal_readiness_candidate"])
        self.assertAlmostEqual(summary["r_plus_omission_anticipatory_response_fraction_0_to_1s"], 0.5)
        self.assertTrue(any("high unrewarded" in warning for warning in summary["warnings"]))
        self.assertEqual(len(summary["roles"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
