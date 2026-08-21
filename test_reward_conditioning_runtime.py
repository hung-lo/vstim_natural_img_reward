#!/usr/bin/env python3
"""Runtime regression tests for reward-conditioning behavior."""

import tempfile
import io
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import PIL  # noqa: F401
except ImportError:
    # Runtime tests use fake screen objects and never render PIL images.
    pil_stub = types.ModuleType("PIL")
    pil_stub.Image = object
    pil_stub.ImageDraw = object
    pil_stub.ImageOps = object
    sys.modules["PIL"] = pil_stub

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

    def trigger_reward(self, context):
        return "reward_1"

    def trigger_suction(self, context):
        return "suction_1"


class RewardConditioningRuntimeTests(unittest.TestCase):
    def test_status_reporter_throttles_and_session_status_table(self):
        class Stream(io.StringIO):
            def isatty(self): return True
        times = iter([0.0, 0.2, 1.0, 2.0])
        stream = Stream()
        reporter = reward.StatusReporter(stream=stream, monotonic_fn=lambda: next(times))
        self.assertTrue(reporter.report("a"))
        self.assertFalse(reporter.report("b"))
        self.assertTrue(reporter.report("c"))
        self.assertTrue(reporter.report("d"))
        cases = [
            ({"task_completed": True, "post_completed": True}, "complete"),
            ({"task_completed": False, "post_completed": False, "interrupted": True}, "interrupted"),
            ({"task_completed": False, "post_completed": False, "primary_error": RuntimeError("x")}, "failed"),
            ({"task_completed": True, "post_completed": True, "camera_enabled": True, "camera_state": {"camera_stop_confirmed": True, "camera_raw_files_verified": True}}, "protocol_complete_video_pending"),
            ({"task_completed": True, "post_completed": True, "cleanup_errors": ["x"]}, "cleanup_failed"),
        ]
        for kwargs, expected in cases:
            self.assertEqual(reward.derive_session_status(**kwargs), expected)

    def test_non_tty_status_reporter_uses_thirty_second_interval(self):
        class Stream(io.StringIO):
            def isatty(self): return False
        times = iter([0.0, 29.9, 30.0])
        stream = Stream()
        reporter = reward.StatusReporter(stream=stream, monotonic_fn=lambda: next(times))
        self.assertTrue(reporter.report("first"))
        self.assertFalse(reporter.report("too soon"))
        self.assertTrue(reporter.report("second"))
        self.assertEqual(stream.getvalue(), "first\nsecond\n")

    def test_phase_status_contains_live_eta_and_gate_has_no_finish(self):
        waiting = reward.format_operator_status(
            "WAITING_FOR_2P", 10.0, 100.0, wall_time_sec=1_000.0)
        self.assertNotIn("finish", waiting.lower())

        pre = reward.format_operator_status(
            "PRE", 11.0, 20.0, protocol_remaining_sec=80.0,
            wall_time_sec=1_000.0)
        self.assertIn("PRE", pre)
        self.assertIn("protocol remaining", pre)
        self.assertIn("finish ~", pre)

        task = reward.format_operator_status(
            "TASK", 12.0, 40.0, 3, 10, 15.0, wall_time_sec=1_000.0)
        self.assertIn("TASK 3/10 (30.0%)", task)
        self.assertIn("+POST", task)
        self.assertIn("finish ~", task)

        post = reward.format_operator_status(
            "POST", 13.0, 10.0, wall_time_sec=1_000.0)
        self.assertIn("POST", post)
        self.assertIn("finish ~", post)

    def test_session_status_explicit_camera_cleanup_error_beats_video_pending(self):
        state = {
            "camera_stop_confirmed": True,
            "camera_raw_files_verified": True,
            "camera_mp4_verified": False,
        }
        self.assertEqual(
            reward.derive_session_status(
                True, True, camera_enabled=True, camera_state=state,
                camera_cleanup_error=True),
            "protocol_complete_camera_cleanup_failed",
        )
        self.assertEqual(
            reward.derive_session_status(
                True, True, camera_enabled=True, camera_state=state,
                camera_cleanup_error=False),
            "protocol_complete_video_pending",
        )
        unsafe_state = dict(state, camera_raw_files_verified=False)
        self.assertEqual(
            reward.derive_session_status(
                True, True, camera_enabled=True, camera_state=unsafe_state),
            "protocol_complete_camera_cleanup_failed",
        )

    def test_strict_prompt_helpers_reprompt_and_reject_eof(self):
        with mock.patch("builtins.input", side_effect=["2.9", "abc", "0", "2"]):
            self.assertEqual(reward.prompt_int_with_default("integer", 1, minimum=1), 2)
        with mock.patch("builtins.input", side_effect=["nan", "inf", "-inf", "bad", "2.5"]):
            self.assertEqual(reward.prompt_float_or_default("float", 1.0), 2.5)
        with mock.patch("builtins.input", side_effect=["maybe", "yes"]):
            self.assertTrue(reward.prompt_yes_no_strict("choice", default_yes=False))
        with mock.patch("builtins.input", return_value=""):
            self.assertFalse(reward.prompt_yes_no_strict("choice", default_yes=False))
        for raw, expected in (("y", True), ("yes", True), ("n", False), ("no", False)):
            with mock.patch("builtins.input", return_value=raw):
                self.assertEqual(
                    reward.prompt_yes_no_strict("choice", default_yes=None),
                    expected,
                )
        with mock.patch("builtins.input", side_effect=EOFError):
            with self.assertRaisesRegex(RuntimeError, "EOF"):
                reward.prompt_int_with_default("integer", 1)
        with mock.patch("builtins.input", side_effect=EOFError):
            with self.assertRaisesRegex(RuntimeError, "EOF"):
                reward.prompt_float_or_default("float", 1.0)
        with mock.patch("builtins.input", side_effect=EOFError):
            with self.assertRaisesRegex(RuntimeError, "EOF"):
                reward.prompt_yes_no_strict("choice", default_yes=False)

    def test_planned_task_remaining_uses_realized_itis_and_skips_final(self):
        trials = [{"planned_iti_duration_sec": 2.0}, {"planned_iti_duration_sec": 7.0}, {"planned_iti_duration_sec": 99.0}]
        self.assertEqual(reward.estimate_task_seconds(trials), 13.5)
        self.assertEqual(reward.planned_task_remaining_seconds(trials, 1), 10.0)
        self.assertEqual(reward.planned_task_remaining_seconds(trials, 3), 0.0)
        self.assertEqual(
            reward.planned_task_remaining_during_iti(
                trials, 0, iti_deadline_monotonic=105.0,
                now_monotonic=101.0),
            14.0,
        )
        repeated = [
            reward.planned_task_remaining_during_iti(
                trials, 0, iti_deadline_monotonic=100.0,
                now_monotonic=100.0 - current_iti_remaining)
            for current_iti_remaining in (3.0, 2.0, 1.0)
        ]
        self.assertEqual(repeated, [13.0, 12.0, 11.0])

    def test_wait_until_services_status_from_existing_poll_loop(self):
        gpio_client = FakeGPIOClient()
        remaining_values = []
        monotonic_values = iter([0.0, 0.4, 0.4, 0.8, 1.0])
        with mock.patch.object(reward.time, "monotonic", side_effect=lambda: next(monotonic_values)), \
             mock.patch.object(reward.time, "sleep"):
            reward.wait_until(
                1.0, gpio_client, Path("unused.csv"), [],
                status_callback=remaining_values.append)
        self.assertEqual(gpio_client.drain_calls, 2)
        self.assertEqual(len(remaining_values), 2)
        self.assertAlmostEqual(remaining_values[0], 0.6)
        self.assertAlmostEqual(remaining_values[1], 0.2)

    def test_two_photon_gate_services_callbacks_and_rejects_eof(self):
        class Stdin(io.StringIO):
            def isatty(self): return True

        status = mock.Mock()
        service = mock.Mock()
        stdin = Stdin("\n")
        with mock.patch.object(reward.sys, "stdin", stdin), \
             mock.patch.object(reward.select, "select", return_value=([stdin], [], [])), \
             mock.patch.object(reward.time, "monotonic", side_effect=[10.0, 10.5]):
            self.assertEqual(
                reward.wait_for_two_photon_gate(status, service, poll_sec=0.0),
                0.5,
            )
        status.assert_called_once()
        service.assert_called_once()

        eof_stdin = Stdin("")
        with mock.patch.object(reward.sys, "stdin", eof_stdin), \
             mock.patch.object(reward.select, "select", return_value=([eof_stdin], [], [])), \
             mock.patch.object(reward.time, "monotonic", return_value=10.0):
            with self.assertRaisesRegex(RuntimeError, "EOF"):
                reward.wait_for_two_photon_gate(poll_sec=0.0)

    def test_cli_camera_options_are_exclusive(self):
        self.assertTrue(reward.parse_args(["--camera"]).camera)
        with self.assertRaises(SystemExit):
            reward.parse_args(["--camera", "--no-camera"])

    def test_latest_camera_state_populates_final_metadata(self):
        fetch_state = {
            "camera_transfer_completed": True, "camera_raw_files_verified": True,
            "camera_raw_hash_verified": True, "camera_conversion_completed": False,
            "camera_mp4_verified": False, "remote_raw_cleanup_completed": False,
            "remote_raw_retained": True,
        }
        convert_state = dict(fetch_state)
        convert_state.update({"camera_conversion_completed": True, "camera_mp4_verified": True,
                              "remote_raw_cleanup_completed": True, "remote_raw_retained": False})
        self.assertFalse(reward.final_camera_metadata(fetch_state, True)["camera_mp4_verified"])
        summary = reward.final_camera_metadata(convert_state, True)
        self.assertTrue(summary["camera_conversion_completed"])
        self.assertTrue(summary["camera_mp4_verified"])
        self.assertTrue(summary["remote_raw_cleanup_completed"])
        self.assertFalse(summary["remote_raw_retained"])

    def test_final_session_exit_preserves_primary_and_interrupts(self):
        self.assertEqual(reward.final_session_exit(None, True, ["cleanup"], []), 130)
        self.assertEqual(
            reward.final_session_exit(
                None, True, [], [], camera_cleanup_error=True),
            130,
        )
        with self.assertRaisesRegex(RuntimeError, "primary experiment failure"):
            reward.final_session_exit(RuntimeError("primary experiment failure"), False, ["cleanup failure"], [])
        with self.assertRaisesRegex(RuntimeError, "Camera cleanup failed"):
            reward.final_session_exit(
                None, False, [], [], camera_cleanup_error=True,
                camera_cleanup_error_message="conversion exploded")
        with self.assertRaisesRegex(RuntimeError, "cleanup failure"):
            reward.final_session_exit(None, False, ["cleanup failure"], [])

    def test_camera_choice_elapsed_and_manifest_status_helpers(self):
        support = object()
        prompt = mock.Mock(return_value=False)
        self.assertTrue(reward.resolve_camera_choice(True, False, support, prompt))
        prompt.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "camera support is unavailable"):
            reward.resolve_camera_choice(True, False, None, prompt)
        self.assertFalse(reward.resolve_camera_choice(False, True, support, prompt))
        prompt.assert_not_called()
        self.assertFalse(reward.resolve_camera_choice(False, False, support, prompt))
        prompt.assert_called_once()
        prompt.reset_mock()
        self.assertFalse(reward.resolve_camera_choice(False, False, None, prompt))
        prompt.assert_not_called()

        elapsed = reward.camera_elapsed_from_ns(100_000_000_000, 160_000_000_000)
        self.assertEqual(elapsed, 60.0)
        self.assertEqual(elapsed, reward.camera_elapsed_from_ns(
            100_000_000_000, 160_000_000_000))

        with tempfile.TemporaryDirectory(prefix="manifest_status_") as temp_dir:
            path = Path(temp_dir) / "session_manifest.json"
            reward.atomic_write_json(path, {"status": "preparing"})
            for status in (
                    "complete", "protocol_complete_video_pending",
                    "protocol_complete_camera_cleanup_failed"):
                manifest = reward.update_session_manifest_status(path, status)
                metadata = {"session_status": status}
                self.assertEqual(metadata["session_status"], manifest["status"])

    def test_setup_summary_contains_operator_decisions(self):
        trials = [
            {"reward_scheduled": True, "reward_omission_scheduled": False,
             "suction_scheduled": True},
            {"reward_scheduled": False, "reward_omission_scheduled": True,
             "suction_scheduled": True},
        ]
        summary = reward.format_setup_summary(
            "mouse1", "notes", Path("assignment.json"), False, 1, trials,
            3.0, 4.5, 300.0, 300.0, 12.0, 612.0,
            {"reward_pin_bcm": 19, "lick_pin_bcm": 26,
             "suction_pin_bcm": 25, "simulate_gpio": True},
            True, 3.5, 0.3, output_root=Path("/output"))
        for text in (
                "Mouse:", "Session notes:", "Blocks:", "Total trials:",
                "Scheduled rewards:", "Scheduled omissions:",
                "Scheduled suction events:", "open-loop reward boundary: 1.0 s",
                "ITI: uniform", "PRE gray background:", "POST gray background:",
                "Planned task duration:", "Planned PRE + task + POST:",
                "Camera/video cleanup not included.", "GPIO simulation:",
                "Face camera:", "Output destination:"):
            self.assertIn(text, summary)

    def test_operational_timing_metadata_fields_and_order(self):
        metadata = {
            "operator_gate_enter_monotonic_ns": 100,
            "operator_gate_release_monotonic_ns": 110,
            "operator_gate_wait_sec": 10e-9,
            "camera_recording_request_monotonic_ns": 90,
            "camera_recording_confirmed_monotonic_ns": 95,
            "camera_stop_confirmed_monotonic_ns": 180,
            "camera_recording_elapsed_local_sec": 90e-9,
            "pre_start_monotonic_ns": 120,
            "pre_end_monotonic_ns": 130,
            "pre_elapsed_sec": 10e-9,
            "task_start_monotonic_ns": 140,
            "task_end_monotonic_ns": 150,
            "post_start_monotonic_ns": 160,
            "post_end_monotonic_ns": 170,
            "post_elapsed_sec": 10e-9,
        }
        self.assertTrue(reward.validate_operational_timing_metadata(
            metadata, camera_required=True))
        broken = dict(metadata, task_end_monotonic_ns=125)
        with self.assertRaisesRegex(ValueError, "out of order"):
            reward.validate_operational_timing_metadata(
                broken, camera_required=True)

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

        status_updates = []

        def fake_wait_until(deadline_monotonic, gpio_client_arg, event_log_path,
                            all_gpio_events, status_callback=None):
            wait_calls.append(deadline_monotonic)
            if status_callback:
                for remaining in (0.2, 0.1, 0.0):
                    with mock.patch.object(
                            reward.time, "monotonic",
                            return_value=deadline_monotonic - remaining):
                        status_callback(remaining)

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
                    status_callback=lambda done, total, remaining: status_updates.append(
                        (done, total, remaining)),
                )

        self.assertEqual(len(wait_calls), 1)
        self.assertTrue(runtime_by_trial[0]["trial_completed"])
        self.assertTrue(runtime_by_trial[1]["trial_completed"])
        self.assertEqual(pending_final_reward_checks, [])
        self.assertEqual(len(status_updates), 3)
        self.assertTrue(all(update[:2] == (1, 2) for update in status_updates))
        self.assertEqual(
            [round(update[2], 6) for update in status_updates],
            [1.7, 1.6, 1.5],
        )

    def test_reward_boundary_order_and_suction_waits_share_iti_status(self):
        def trial(index, reward_scheduled=False, suction_scheduled=False):
            return {
                "trial_index": index, "trial_number": index + 1,
                "block_number": 1, "image_role": "role_%d" % index,
                "image_category": "conditioned", "image_id": index + 1,
                "image_filename": "img_%d.png" % index,
                "reward_eligible": reward_scheduled,
                "reward_scheduled": reward_scheduled,
                "reward_omission_scheduled": False,
                "suction_scheduled": suction_scheduled,
                "planned_iti_duration_sec": 4.0 if index == 0 else 9.0,
            }

        trials = [trial(0, True, True), trial(1)]
        loaded_raws = {
            item["image_filename"]: {"first": Path("first.raw"), "second": Path("second.raw")}
            for item in trials
        }
        raw_paths = {
            item["image_filename"]: {"first": Path("first.raw"), "second": Path("second.raw")}
            for item in trials
        }
        calls = []
        display_count = {"value": 0}

        def fake_display(screen, raw_path):
            labels = ("segment1", "segment2", "background")
            label = labels[display_count["value"] % 3]
            calls.append(label)
            display_count["value"] += 1
            base_ns = display_count["value"] * 1_000_000_000
            return SimpleNamespace(start_time=0, mean_interframe=0, stddev_interframe=0), {
                "request_utc_iso": "iso", "request_unix_sec": "1.0",
                "request_unix_ns": base_ns,
                "request_perf_counter_ns": base_ns,
                "return_unix_ns": base_ns + 1,
                "return_perf_counter_ns": base_ns + 1,
                "duration_sec": 0.001,
            }

        gpio = FakeGPIOClient()
        gpio.trigger_reward = mock.Mock(side_effect=lambda context: calls.append("reward") or "reward_1")
        gpio.trigger_suction = mock.Mock(side_effect=lambda context: calls.append("suction") or "suction_1")

        def fake_wait(deadline, gpio_client, event_path, events, status_callback=None):
            calls.append("wait_with_status" if status_callback else "wait_without_status")
            if status_callback:
                with mock.patch.object(reward.time, "monotonic", return_value=deadline - 0.5):
                    status_callback(0.5)

        status_updates = []
        with tempfile.TemporaryDirectory(prefix="reward_boundary_") as temp_dir, \
             mock.patch.object(reward.base, "display_raw_with_timing", side_effect=fake_display), \
             mock.patch.object(reward, "wait_until", side_effect=fake_wait), \
             mock.patch.object(reward, "verify_reward_command", side_effect=lambda *args, **kwargs: calls.append("verify_reward")), \
             mock.patch.object(reward, "verify_suction_command", side_effect=lambda *args, **kwargs: calls.append("verify_suction")):
            reward.run_trials(
                SimpleNamespace(), trials, loaded_raws, Path("gray.loaded"),
                raw_paths, Path(temp_dir) / "gray.raw", gpio,
                Path(temp_dir) / "events.csv", [], 6, 0.5, 3.5, 0.05,
                status_callback=lambda done, total, remaining: (
                    calls.append("status"),
                    status_updates.append((done, total, remaining))),
            )

        self.assertEqual(calls[:4], ["segment1", "reward", "segment2", "background"])
        self.assertNotIn("status", calls[:4])
        self.assertEqual(calls.count("wait_with_status"), 2)
        self.assertEqual(calls.count("status"), 2)
        self.assertLess(calls.index("wait_with_status"), calls.index("suction"))
        self.assertLess(calls.index("suction"), calls.index("verify_suction"))
        self.assertTrue(all(update[:2] == (1, 2) for update in status_updates))


if __name__ == "__main__":
    unittest.main(verbosity=2)
