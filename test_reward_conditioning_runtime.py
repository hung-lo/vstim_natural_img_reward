#!/usr/bin/env python3
"""Runtime regression tests for reward-conditioning behavior."""

import tempfile
import io
import json
import subprocess
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
    def test_display_timing_calibration_falls_back_to_nominal_refreshes(self):
        resolved = reward.resolve_display_timing_calibration({})
        self.assertFalse(resolved["calibration_enabled"])
        self.assertIsNone(resolved["calibration_id"])
        self.assertEqual(resolved["calibration_refresh_rate_hz"], 60.0)
        self.assertEqual(resolved["segment1_refreshes"], 60)
        self.assertEqual(resolved["segment2_refreshes"], 30)
        self.assertEqual(resolved["stim_total_refreshes"], 90)
        self.assertEqual(resolved["stimulus_onset_compensation_sec"], 0.0)

    def test_box151_display_timing_calibration_resolves_without_rescaling(self):
        resolved = reward.resolve_display_timing_calibration({
            "display_timing_calibration_id": "box151_photodiode_20ksps_50trial_60hz_v1",
            "display_timing_calibration_refresh_rate_hz": 60.0,
            "stim_segment1_refreshes": 64,
            "stim_segment2_refreshes": 24,
            "stimulus_onset_compensation_sec": 0.095,
        })
        self.assertTrue(resolved["calibration_enabled"])
        self.assertEqual(resolved["segment1_refreshes"], 64)
        self.assertEqual(resolved["segment2_refreshes"], 24)
        self.assertAlmostEqual(resolved["segment1_programmed_duration_sec"], 64.0 / 60.0)
        self.assertAlmostEqual(resolved["segment2_programmed_duration_sec"], 24.0 / 60.0)
        self.assertAlmostEqual(resolved["total_programmed_duration_sec"], 88.0 / 60.0)
        self.assertAlmostEqual(resolved["stimulus_onset_compensation_sec"], 0.095)

    def test_display_timing_calibration_rejects_refresh_rate_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            reward.resolve_display_timing_calibration({
                "display_timing_calibration_id": "box151",
                "display_timing_calibration_refresh_rate_hz": 59.94,
                "stim_segment1_refreshes": 64,
                "stim_segment2_refreshes": 24,
                "stimulus_onset_compensation_sec": 0.095,
            })

    def test_suction_target_applies_compensation_without_changing_behavioral_delay(self):
        self.assertAlmostEqual(
            reward.suction_target_from_stim_request(100_000_000_000, 3.4, 0.095),
            103.495,
        )
        self.assertAlmostEqual(
            reward.suction_target_from_stim_request(100_000_000_000, 3.4, 0.0),
            103.4,
        )

    def test_explicit_refresh_conversion_passes_refresh_count_to_rpg(self):
        import run_stringer_vstim as vstim

        class FakeRPG:
            def __init__(self):
                self.calls = []

            def convert_raw(self, *args):
                self.calls.append(args)

        with tempfile.TemporaryDirectory(prefix="refresh_conversion_") as temp_dir:
            raw_path = Path(temp_dir) / "stim_reward_pre_64r.raw"
            rpg_module = FakeRPG()
            canvas = SimpleNamespace(
                convert=lambda mode: SimpleNamespace(tobytes=lambda: b"rgb")
            )
            vstim.convert_canvas_to_rpg_raw_refreshes(rpg_module, canvas, raw_path, 64)
            self.assertEqual(rpg_module.calls[0][5], 64)
            self.assertTrue(raw_path.parent.exists())

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
            ({"task_completed": True, "post_completed": True, "camera_enabled": True, "camera_state": {"camera_stop_confirmed": True, "camera_raw_files_verified": True, "camera_raw_hash_verified": True}}, "protocol_complete_video_pending"),
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
            "camera_raw_hash_verified": True,
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

    def test_simulate_water_test_requires_simulate_gpio(self):
        with self.assertRaises(SystemExit):
            reward.parse_args(["--simulate-water-test"])
        args = reward.parse_args(["--simulate-gpio", "--simulate-water-test"])
        self.assertTrue(args.simulate_gpio)
        self.assertTrue(args.simulate_water_test)

    def test_simulate_behavior_test_requires_simulate_gpio_and_composes(self):
        with self.assertRaises(SystemExit):
            reward.parse_args(["--simulate-behavior-test"])
        args = reward.parse_args([
            "--simulate-gpio", "--simulate-behavior-test", "--simulate-water-test"
        ])
        self.assertTrue(args.simulate_behavior_test)
        self.assertTrue(args.simulate_water_test)

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

        elapsed = reward.camera_elapsed_from_ns(107_000_000_000, 377_000_000_000)
        self.assertEqual(elapsed, 270.0)
        self.assertEqual(elapsed, reward.camera_elapsed_from_ns(
            107_000_000_000, 377_000_000_000))
        conversion_completed_ns = 450_000_000_000
        self.assertEqual(
            elapsed,
            reward.camera_elapsed_from_ns(
                107_000_000_000, 377_000_000_000
            ),
        )
        self.assertGreater(conversion_completed_ns, 377_000_000_000)
        self.assertEqual(
            reward.camera_startup_confirmation_latency_from_ns(
                100_000_000_000, 107_000_000_000
            ),
            7.0,
        )
        self.assertIsNone(reward.camera_elapsed_from_ns(None, 377_000_000_000))
        self.assertIsNone(
            reward.camera_startup_confirmation_latency_from_ns(
                100_000_000_000, None
            )
        )

        with tempfile.TemporaryDirectory(prefix="manifest_status_") as temp_dir:
            path = Path(temp_dir) / "session_manifest.json"
            reward.atomic_write_json(path, {"status": "preparing"})
            for status in (
                    "complete", "protocol_complete_video_pending",
                    "protocol_complete_camera_cleanup_failed"):
                manifest = reward.update_session_manifest_status(path, status)
                metadata = {"session_status": status}
                self.assertEqual(metadata["session_status"], manifest["status"])

            session_root = Path(temp_dir)
            video_manifest = session_root / "video" / "video_manifest.json"
            video_manifest.parent.mkdir()
            video_manifest.write_text("{}")
            camera_event_log = session_root / "video" / "camera_control_events.csv"
            absent_artifact = reward.build_session_manifest(
                session_root, "session", "mouse", "protocol", "complete", {
                    "camera_event_log": camera_event_log,
                })
            self.assertFalse(absent_artifact["files"]["camera_event_log"]["exists"])
            camera_event_log.write_text("event,details\n")
            artifact = reward.build_session_manifest(
                session_root, "session", "mouse", "protocol", "complete", {
                    "metadata": session_root / "metadata.json",
                    "video_manifest": video_manifest,
                    "camera_event_log": camera_event_log,
                })
            self.assertFalse(artifact["files"]["metadata"]["exists"])
            self.assertTrue(artifact["files"]["video_manifest"]["exists"])
            self.assertTrue(artifact["files"]["camera_event_log"]["exists"])
            self.assertEqual(
                artifact["files"]["camera_event_log"]["path"],
                "video/camera_control_events.csv",
            )

            disabled_artifact = reward.build_session_manifest(
                session_root, "session", "mouse", "protocol", "complete", {
                    "camera_event_log": None,
                })
            self.assertFalse(disabled_artifact["files"]["camera_event_log"]["exists"])
            self.assertIsNone(disabled_artifact["files"]["camera_event_log"]["path"])

    def test_reward_volume_config_validation_and_tracker_accounting(self):
        config = reward.load_reward_volume_config({
            "reward_volume_ul_per_train": "10",
            "maximum_session_reward_ul": 30,
        })
        self.assertTrue(config["reward_volume_cap_enabled"])
        for name in ("reward_volume_ul_per_train", "maximum_session_reward_ul"):
            for value in (float("nan"), float("inf"), -1, 0, "bad"):
                with self.assertRaisesRegex(RuntimeError, name):
                    reward.load_reward_volume_config({
                        "reward_volume_ul_per_train": 10,
                        "maximum_session_reward_ul": 30,
                        name: value,
                    })
        disabled = reward.load_reward_volume_config({
            "reward_volume_ul_per_train": None,
            "maximum_session_reward_ul": None,
        })
        self.assertFalse(disabled["reward_volume_cap_enabled"])
        with self.assertRaisesRegex(RuntimeError, "requires"):
            reward.load_reward_volume_config({
                "reward_volume_ul_per_train": None,
                "maximum_session_reward_ul": 30,
            })

        trials = [
            {"reward_scheduled": True},
            {"reward_scheduled": False, "reward_omission_scheduled": True},
            {"reward_scheduled": True},
        ]
        tracker = reward.RewardVolumeTracker(10.0, 30.0,
                                             reward.planned_reward_train_count(trials))
        self.assertEqual(tracker.planned_reward_volume_ul, 20.0)
        tracker.preflight()
        tracker.check_next_command(scheduled=False)
        tracker.record_dispatched(manual=True)
        tracker.check_next_command(scheduled=True)
        tracker.record_dispatched(scheduled=True)
        self.assertEqual(tracker.manual_reward_train_count, 1)
        self.assertEqual(tracker.delivered_reward_train_count, 2)
        self.assertEqual(tracker.estimated_delivered_reward_ul, 20.0)
        with self.assertRaises(reward.RewardVolumeCapExceeded):
            tracker.record_dispatched(scheduled=True)
            tracker.check_next_command(scheduled=True)
        self.assertTrue(tracker.reward_volume_cap_exceeded)

        exact = reward.RewardVolumeTracker(10.0, 20.0, 2)
        exact.preflight()
        exact.check_next_command(scheduled=True)
        exact.record_dispatched(scheduled=True)
        exact.check_next_command(scheduled=True)
        exact.record_dispatched(scheduled=True)
        self.assertEqual(exact.estimated_delivered_reward_ul, 20.0)
        with self.assertRaises(reward.RewardVolumeCapExceeded):
            reward.RewardVolumeTracker(10.0, 15.0, 2).preflight()

    def test_box151_reward_train_may_extend_safely_into_iti(self):
        hardware = {
            "reward_pin_bcm": 19,
            "lick_pin_bcm": 26,
            "suction_pin_bcm": 25,
            "reward_num_pulses": 6,
            "reward_pulse_on_sec": 0.1,
            "reward_pulse_off_sec": 0.01,
            "suction_delay_from_stim_onset_sec": 3.4,
            "suction_duration_sec": 0.25,
            "reward_volume_ul_per_train": None,
            "maximum_session_reward_ul": None,
        }
        with tempfile.TemporaryDirectory(prefix="box151_reward_timing_") as temp_dir:
            config_path = Path(temp_dir) / "hardware.json"
            reward.atomic_write_json(config_path, hardware)
            loaded = reward.load_hardware_config(config_path)

            timing = reward.calculate_reward_timing(loaded)
            self.assertAlmostEqual(timing["reward_train_duration_sec"], 0.65)
            self.assertAlmostEqual(
                timing["reward_nominal_start_from_stim_onset_sec"], 1.0
            )
            self.assertAlmostEqual(
                timing["reward_nominal_completion_from_stim_onset_sec"], 1.65
            )
            self.assertAlmostEqual(
                timing["reward_train_extension_into_iti_sec"], 0.15
            )
            self.assertAlmostEqual(
                timing["reward_to_suction_nominal_gap_sec"], 1.75
            )
            self.assertTrue(timing["reward_train_extends_into_iti"])

            hardware["suction_duration_sec"] = 0.36
            reward.atomic_write_json(config_path, hardware)
            with self.assertRaisesRegex(RuntimeError, "clean gray"):
                reward.load_hardware_config(config_path)

    def test_reward_timing_accepts_safe_long_exact_and_short_trains(self):
        cases = (
            ("safe_long", 0.6, 3.0, True, 0.1),
            ("exact_segment", 0.5, 3.0, False, 0.0),
            ("short", 0.3, 3.0, False, 0.0),
        )
        for case_name, duration, suction_delay, extends, extension in cases:
            with self.subTest(case=case_name):
                timing = reward.calculate_reward_timing({
                    "reward_num_pulses": 1,
                    "reward_pulse_on_sec": duration,
                    "reward_pulse_off_sec": 0.0,
                    "suction_delay_from_stim_onset_sec": suction_delay,
                })
                self.assertAlmostEqual(
                    timing["reward_train_duration_sec"], duration
                )
                self.assertEqual(
                    timing["reward_train_extends_into_iti"], extends
                )
                self.assertAlmostEqual(
                    timing["reward_train_extension_into_iti_sec"], extension
                )
        self.assertEqual(reward.STIM_DURATION_SEC, 1.5)
        self.assertEqual(reward.REWARD_DELAY_SEC, 1.0)
        self.assertEqual(reward.POST_REWARD_STIM_SEC, 0.5)

    def test_reward_suction_overlap_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "must not overlap"):
            reward.calculate_reward_timing({
                "reward_num_pulses": 1,
                "reward_pulse_on_sec": 2.5,
                "reward_pulse_off_sec": 0.0,
                "suction_delay_from_stim_onset_sec": 3.4,
            })

    def test_main_reward_suction_overlap_aborts_before_hardware_start(self):
        hardware = {
            "reward_num_pulses": 1,
            "reward_pulse_on_sec": 2.5,
            "reward_pulse_off_sec": 0.0,
            "suction_delay_from_stim_onset_sec": 3.4,
        }
        gpio_constructor = mock.Mock()
        camera_import = mock.Mock()
        resolve_image_dir = mock.Mock()
        with mock.patch.dict(sys.modules, {"rpg": types.ModuleType("rpg")}), \
             mock.patch.object(reward.base, "print_environment"), \
             mock.patch.object(reward, "load_hardware_config", return_value=hardware), \
             mock.patch.object(reward, "maybe_import_camera_support", camera_import), \
             mock.patch.object(reward.base, "resolve_image_dir", resolve_image_dir), \
             mock.patch.object(reward, "BehaviorGPIOClient", gpio_constructor):
            with self.assertRaisesRegex(RuntimeError, "must not overlap"):
                reward.main(["--no-camera"])
        camera_import.assert_not_called()
        resolve_image_dir.assert_not_called()
        gpio_constructor.assert_not_called()

    def test_main_planned_over_cap_aborts_before_session_or_hardware(self):
        hardware = {
            "reward_num_pulses": 1,
            "reward_pulse_on_sec": 0.01,
            "reward_pulse_off_sec": 0.0,
            "suction_delay_from_stim_onset_sec": 3.5,
            "suction_duration_sec": 0.05,
            "reward_volume_ul_per_train": 10.0,
            "maximum_session_reward_ul": 5.0,
            "reward_volume_cap_enabled": True,
        }
        trials = [{"reward_scheduled": True}]
        gpio_constructor = mock.Mock()
        ensure_dir = mock.Mock()
        with mock.patch.dict(sys.modules, {"rpg": types.ModuleType("rpg")}), \
             mock.patch.object(reward.base, "print_environment"), \
             mock.patch.object(reward, "load_hardware_config", return_value=hardware), \
             mock.patch.object(reward.base, "resolve_image_dir", return_value=Path("images")), \
             mock.patch.object(reward.base, "list_png_files", return_value=[Path("image.png")]), \
             mock.patch.object(reward, "create_or_load_assignment",
                               return_value=([{"image_filename": "image.png"}],
                                             Path("assignment.json"), False, 1)), \
             mock.patch.object(reward, "make_trial_plan", return_value=(trials, 2)), \
             mock.patch.object(reward, "BehaviorGPIOClient", gpio_constructor), \
             mock.patch.object(reward.base, "ensure_dir", ensure_dir):
            with self.assertRaises(reward.RewardVolumeCapExceeded):
                reward.main([
                    "--no-camera", "--simulate-gpio", "--mouse-id", "mouse",
                    "--session-notes", "test", "--blocks", "1",
                    "--iti-min-sec", "3", "--iti-max-sec", "4",
                    "--pre-background-min", "0", "--post-background-min", "0",
                ])
        gpio_constructor.assert_not_called()
        ensure_dir.assert_not_called()

    def test_canonical_camera_security_and_completion_invariants(self):
        self.assertTrue(reward.derive_camera_data_secured(False, {}))
        secure = {
            "camera_stop_confirmed": True,
            "camera_raw_files_verified": True,
            "camera_raw_hash_verified": True,
            "camera_mp4_verified": True,
            "remote_raw_cleanup_completed": True,
        }
        self.assertTrue(reward.derive_camera_data_secured(True, secure))
        for field in ("camera_raw_hash_verified", "camera_mp4_verified",
                      "camera_stop_confirmed", "remote_raw_cleanup_completed"):
            state = dict(secure, **{field: False})
            self.assertFalse(reward.derive_camera_data_secured(True, state))
        self.assertFalse(reward.derive_camera_data_secured(True, secure, True))

        cases = [
            ({"task_completed": True, "post_background_completed": True},
             True, "complete"),
            ({"task_completed": True, "post_background_completed": True,
              "camera_enabled": True, "camera_state": secure},
             True, "complete"),
            ({"task_completed": True, "post_background_completed": True,
              "camera_enabled": True,
              "camera_state": dict(secure, camera_mp4_verified=False,
                                   remote_raw_cleanup_completed=False)},
             False, "protocol_complete_video_pending"),
            ({"task_completed": True, "post_background_completed": True,
              "camera_enabled": True, "camera_state": secure,
              "camera_cleanup_error": True},
             False, "protocol_complete_camera_cleanup_failed"),
            ({"task_completed": True, "post_background_completed": True,
              "finalization_errors": ["manifest failed"]},
             False, "cleanup_failed"),
            ({"task_completed": False, "post_background_completed": False,
              "primary_error": RuntimeError("primary")},
             False, "failed"),
            ({"task_completed": False, "post_background_completed": False,
              "interrupted": True},
             False, "interrupted"),
        ]
        for kwargs, expected_completed, expected_status in cases:
            state = reward.resolve_final_completion_state(**kwargs)
            self.assertEqual(state["session_completed"], expected_completed)
            self.assertEqual(state["session_status"], expected_status)
            self.assertEqual(
                state["session_completed"],
                state["session_status"] == "complete")

    def test_final_status_artifact_repair_pass(self):
        completion = {
            "task_completed": True,
            "post_background_completed": True,
            "camera_enabled": False,
        }

        def exercise(failing_name=None, primary_error=None):
            with tempfile.TemporaryDirectory(prefix="final_status_") as temp_dir:
                root = Path(temp_dir)
                metadata_path = root / "metadata.json"
                manifest_path = root / "session_manifest.json"
                reward.atomic_write_json(metadata_path, {"session_status": "preparing"})
                errors = []

                def writer(path, payload):
                    if Path(path).name == failing_name:
                        raise OSError("injected %s failure" % failing_name)
                    reward.atomic_write_json(path, payload)

                inputs = dict(completion)
                inputs["primary_error"] = primary_error
                state = reward.finalize_status_artifacts(
                    {"protocol": "protocol"}, metadata_path, manifest_path,
                    root, "session", "mouse", "protocol",
                    {"metadata": metadata_path}, inputs, [], errors,
                    write_json_fn=writer)
                metadata = json.loads(metadata_path.read_text())
                manifest = (json.loads(manifest_path.read_text())
                            if manifest_path.exists() else None)
                return state, errors, metadata, manifest

        clean, errors, metadata, manifest = exercise()
        self.assertEqual(clean["session_status"], "complete")
        self.assertTrue(clean["session_completed"])
        self.assertEqual(errors, [])
        self.assertEqual(metadata["session_status"], manifest["status"])

        state, errors, metadata, manifest = exercise("session_manifest.json")
        self.assertFalse(state["session_completed"])
        self.assertEqual(metadata["session_status"], "cleanup_failed")
        self.assertTrue(any("manifest:" in error for error in errors))
        with self.assertRaises(RuntimeError):
            reward.final_session_exit(None, False, [], errors)

        state, errors, metadata, manifest = exercise("metadata.json")
        self.assertEqual(state["session_status"], "cleanup_failed")
        self.assertEqual(manifest["status"], "cleanup_failed")
        self.assertTrue(manifest["files"]["metadata"]["exists"])

        primary = RuntimeError("primary experiment failure")
        state, errors, metadata, manifest = exercise(
            "session_manifest.json", primary_error=primary)
        self.assertEqual(state["session_status"], "failed")
        self.assertEqual(metadata["session_status"], "failed")
        with self.assertRaisesRegex(RuntimeError, "primary experiment failure"):
            reward.final_session_exit(primary, False, [], errors)

    def test_stale_entrypoints_fail_before_side_effects(self):
        for script in ("run_stringer_vstim.py", "run_stringer_vstim_cam.py"):
            result = subprocess.run(
                [sys.executable, script], cwd=Path(__file__).parent,
                capture_output=True, text=True,
            )
            output = result.stdout + result.stderr
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("run_stringer_reward_conditioning.py", output)
            self.assertIn("vstim_natural", output)
        import run_stringer_vstim
        self.assertTrue(hasattr(run_stringer_vstim, "display_raw_with_timing"))

    def test_setup_summary_contains_operator_decisions(self):
        trials = [
            {"reward_scheduled": True, "reward_omission_scheduled": False,
             "suction_scheduled": True},
            {"reward_scheduled": False, "reward_omission_scheduled": True,
             "suction_scheduled": True},
        ]
        reward_timing = reward.calculate_reward_timing({
            "reward_num_pulses": 6,
            "reward_pulse_on_sec": 0.1,
            "reward_pulse_off_sec": 0.01,
            "suction_delay_from_stim_onset_sec": 3.4,
        })
        summary = reward.format_setup_summary(
            "mouse1", "notes", Path("assignment.json"), False, 1, trials,
            3.0, 4.5, 300.0, 300.0, 12.0, 612.0,
            {"reward_pin_bcm": 19, "lick_pin_bcm": 26,
             "suction_pin_bcm": 25, "simulate_gpio": True},
            True, 3.4, 0.65, reward_timing=reward_timing,
            output_root=Path("/output"))
        for text in (
                "Mouse:", "Session notes:", "Blocks:", "Total trials:",
                "Scheduled rewards:", "Scheduled omissions:",
                "Scheduled suction events:", "open-loop reward boundary: 1.0 s",
                "ITI: uniform", "PRE gray background:", "POST gray background:",
                "Planned task duration:", "Planned PRE + task + POST:",
                "Camera/video cleanup not included.", "GPIO simulation:",
                "Synthetic water-accounting test:",
                "Reward pulse-train duration: 0.650 s",
                "Reward nominal completion: 1.650 s",
                "Reward extends into gray ITI: yes (0.150 s)",
                "Suction onset: 3.400 s",
                "Reward-to-suction nominal gap: 1.750 s",
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

    def test_trial_summary_reports_physical_onset_suction_estimate(self):
        trial = {
            "trial_index": 0, "trial_number": 1, "block_number": 1,
            "image_role": "rewarded_high_1", "image_category": "conditioned",
            "image_id": 1, "image_filename": "img.png",
            "reward_eligible": True, "reward_scheduled": True,
            "reward_omission_scheduled": False, "suction_scheduled": True,
            "planned_iti_duration_sec": 4.0,
        }
        rows = reward.build_trial_summary(
            [trial],
            {0: {
                "trial_executed": True,
                "stim_presented": True,
                "trial_completed": True,
                "stim_request_monotonic_ns": 100_000_000_000,
                "suction_command_id": "suction_1",
            }},
            [{"command_id": "suction_1", "event_type": "suction_on",
              "unix_time_ns": 103_495_000_000,
              "monotonic_ns": 103_495_000_000}],
            stimulus_onset_compensation_sec=0.095,
        )
        self.assertAlmostEqual(rows[0]["software_suction_delay_sec"], 3.495)
        self.assertAlmostEqual(
            rows[0]["estimated_suction_delay_from_physical_onset_sec"], 3.4
        )

    def test_telemetry_reward_fields_and_physical_lick_alignment(self):
        trial = {
            "trial_number": 24, "block_number": 1,
            "image_filename": "natimg_center_1825.png",
            "image_role": "rewarded_high_1",
            "reward_scheduled": True,
            "reward_omission_scheduled": False,
        }
        runtime = {
            "reward_command_id": "reward_1",
            "suction_command_id": "suction_1",
            "stim_request_monotonic_ns": 100_000_000_000,
        }
        events = [
            {"command_id": "reward_1", "event_type": "reward_command_received"},
            {"command_id": "reward_1", "event_type": "reward_valve_on",
             "monotonic_ns": 101_000_000_000},
            {"command_id": "reward_1", "event_type": "reward_valve_off",
             "monotonic_ns": 101_100_000_000},
            {"command_id": "reward_1", "event_type": "reward_complete"},
            {"command_id": "suction_1", "event_type": "suction_on",
             "monotonic_ns": 103_400_000_000},
            {"event_type": "lick_onset", "monotonic_ns": 100_100_000_000},
            {"event_type": "lick_onset", "monotonic_ns": 101_500_000_000},
            {"event_type": "lick_onset", "monotonic_ns": 104_000_000_000},
        ]
        payload = reward.build_trial_telemetry_payload(
            trial, runtime, events, 500, 10, 1,
            stimulus_onset_compensation_sec=0.1,
        )
        self.assertTrue(payload["reward_delivered"])
        self.assertTrue(payload["reward_contacted"])
        self.assertTrue(payload["anticipatory_lick"])
        self.assertEqual(payload["lick_time_reference"], "estimated_physical_stim_onset")
        self.assertEqual(payload["lick_times_sec"], [0.0, 1.4, 3.9])

    def test_telemetry_reward_contact_is_null_when_required_timing_missing(self):
        trial = {
            "reward_scheduled": True, "reward_omission_scheduled": False,
        }
        events = [
            {"command_id": "reward_1", "event_type": "reward_command_received"},
            {"command_id": "reward_1", "event_type": "reward_valve_on",
             "monotonic_ns": 100_000_000_001},
            {"command_id": "reward_1", "event_type": "reward_valve_off",
             "monotonic_ns": 100_000_000_002},
            {"command_id": "reward_1", "event_type": "reward_complete"},
        ]
        fields = reward.derive_telemetry_reward_fields(
            trial, {"reward_command_id": "reward_1", "suction_command_id": "suction_1",
                    "stim_request_monotonic_ns": 100_000_000_000},
            events, 1,
        )
        self.assertTrue(fields["reward_delivered"])
        self.assertIsNone(fields["reward_contacted"])

    def test_telemetry_reward_contact_is_false_when_known_window_has_no_lick(self):
        trial = {"reward_scheduled": True, "reward_omission_scheduled": False}
        events = [
            {"command_id": "reward_1", "event_type": "reward_command_received"},
            {"command_id": "reward_1", "event_type": "reward_valve_on",
             "monotonic_ns": 101_000_000_000},
            {"command_id": "reward_1", "event_type": "reward_valve_off",
             "monotonic_ns": 101_100_000_000},
            {"command_id": "reward_1", "event_type": "reward_complete"},
            {"command_id": "suction_1", "event_type": "suction_on",
             "monotonic_ns": 103_400_000_000},
            {"event_type": "lick_onset", "monotonic_ns": 100_500_000_000},
            {"event_type": "lick_onset", "monotonic_ns": 103_500_000_000},
        ]
        fields = reward.derive_telemetry_reward_fields(
            trial,
            {"reward_command_id": "reward_1", "suction_command_id": "suction_1",
             "stim_request_monotonic_ns": 100_000_000_000},
            events,
            1,
        )
        self.assertTrue(fields["reward_delivered"])
        self.assertFalse(fields["reward_contacted"])

    def test_simulated_lick_contacts_reward_but_not_omission(self):
        reward_trial = {"reward_scheduled": True, "reward_omission_scheduled": False}
        omission_trial = {"reward_scheduled": False, "reward_omission_scheduled": True}
        reward_runtime = {
            "reward_command_id": "reward_1", "suction_command_id": "suction_1",
            "stim_request_monotonic_ns": 100_000_000_000,
        }
        reward_events = [
            {"command_id": "reward_1", "event_type": "reward_command_received"},
            {"command_id": "reward_1", "event_type": "reward_valve_on",
             "monotonic_ns": 101_000_000_000},
            {"command_id": "reward_1", "event_type": "reward_valve_off",
             "monotonic_ns": 101_100_000_000},
            {"command_id": "reward_1", "event_type": "reward_complete"},
            {"command_id": "suction_1", "event_type": "suction_on",
             "monotonic_ns": 103_400_000_000},
            {"event_type": "lick_onset", "lick_edge": "simulated_test",
             "notes": "synthetic_water_accounting_validation",
             "monotonic_ns": 101_500_000_000},
        ]
        contacted = reward.derive_telemetry_reward_fields(
            reward_trial, reward_runtime, reward_events, 1)
        self.assertTrue(contacted["reward_delivered"])
        self.assertTrue(contacted["reward_contacted"])

        omission_events = [{
            "event_type": "lick_onset", "lick_edge": "simulated_test",
            "notes": "synthetic_water_accounting_validation",
            "monotonic_ns": 101_500_000_000,
        }]
        omission = reward.derive_telemetry_reward_fields(
            omission_trial, {"stim_request_monotonic_ns": 100_000_000_000},
            omission_events, 1)
        self.assertFalse(omission["reward_delivered"])
        self.assertFalse(omission["reward_contacted"])

    def test_telemetry_recent_event_view_excludes_prior_session_events(self):
        events = [
            {"event_type": "lick_onset", "monotonic_ns": 1},
            {"event_type": "lick_onset", "monotonic_ns": 99_400_000_000},
            {"event_type": "lick_onset", "monotonic_ns": 99_600_000_000},
            {"event_type": "lick_onset", "monotonic_ns": 100_100_000_000},
        ]
        recent = reward.recent_trial_gpio_events(events, 100_000_000_000)
        self.assertEqual(
            [event["monotonic_ns"] for event in recent],
            [99_600_000_000, 100_100_000_000],
        )

    def test_telemetry_state_and_trial_failures_are_best_effort(self):
        class FailingReporter:
            publisher = None
            parent_error_count = 0

            def report_state(self, payload, force=False):
                raise RuntimeError("telemetry test failure")

        self.assertFalse(reward.safe_report_telemetry_state(
            FailingReporter(), "PRE", total_trials=1, total_blocks=1))

        class FailingPublisher:
            parent_error_count = 0

            def publish_trial(self, payload):
                raise RuntimeError("telemetry test failure")

        result = reward.publish_completed_trial_telemetry(
            FailingPublisher(),
            {"trial_index": 0, "reward_scheduled": False,
             "reward_omission_scheduled": False},
            {"stim_request_monotonic_ns": 100}, [], 1, 1, 1,
        )
        self.assertIsNone(result)

    def test_task_water_accounting_is_idempotent_and_task_only(self):
        accounting = reward.TaskWaterTelemetryAccounting(3.0)
        accounting.record_trial(1, True, True)
        accounting.record_trial(1, True, True)
        accounting.record_trial(2, True, False)
        accounting.record_trial(3, False, False)
        accounting.record_trial(4, False, None)
        self.assertEqual(accounting.summary()["task_reward_trains_verified_session"], 2)
        self.assertEqual(accounting.summary()["task_reward_trains_contacted_session"], 1)
        self.assertEqual(accounting.summary()["task_water_delivered_ul_session"], 6.0)
        self.assertEqual(accounting.summary()["task_water_likely_consumed_ul_session"], 3.0)
        self.assertIsNone(
            reward.TaskWaterTelemetryAccounting(None).summary()["task_water_delivered_ul_session"]
        )

    def test_behavior_counters_use_explicit_protocol_role_categories(self):
        accounting = reward.TaskWaterTelemetryAccounting(3.0)
        rewarded_high_lick = {"trial_index": 1, "image_role": "rewarded_high_1"}
        rewarded_high_omission = {"trial_index": 2, "image_role": "rewarded_high_2"}
        rewarded_high_no_lick = {"trial_index": 3, "image_role": "rewarded_high_1"}
        unrewarded_high_lick = {"trial_index": 4, "image_role": "unrewarded_high_1"}
        unrewarded_high_no_lick = {"trial_index": 5, "image_role": "unrewarded_high_2"}
        low_lick = {"trial_index": 6, "image_role": "low_01"}
        low_no_lick = {"trial_index": 7, "image_role": "low_02"}

        accounting.record_completed_behavior_trial(rewarded_high_lick, True)
        accounting.record_completed_behavior_trial(rewarded_high_omission, True)
        accounting.record_completed_behavior_trial(rewarded_high_no_lick, False)
        accounting.record_completed_behavior_trial(unrewarded_high_lick, True)
        accounting.record_completed_behavior_trial(unrewarded_high_no_lick, False)
        accounting.record_completed_behavior_trial(low_lick, True)
        accounting.record_completed_behavior_trial(low_no_lick, False)
        # Multiple licks on one completed trial still contribute one numerator.
        accounting.record_completed_behavior_trial(rewarded_high_lick, True)

        summary = accounting.summary()
        self.assertEqual(summary["task_rewarded_high_cue_trials_completed_session"], 3)
        self.assertEqual(summary["task_rewarded_high_cue_anticipatory_lick_trials_session"], 2)
        self.assertEqual(summary["task_unrewarded_high_cue_trials_completed_session"], 2)
        self.assertEqual(summary["task_unrewarded_high_cue_anticipatory_lick_trials_session"], 1)
        self.assertEqual(summary["task_low_probability_cue_trials_completed_session"], 2)
        self.assertEqual(summary["task_low_probability_cue_anticipatory_lick_trials_session"], 1)
        self.assertEqual(
            reward.build_telemetry_state_payload(
                "ITI", total_trials=7, water_accounting=accounting
            )["task_rewarded_high_cue_trials_completed_session"],
            3,
        )
        trial_payload = reward.build_trial_telemetry_payload(
            rewarded_high_no_lick,
            {"stim_request_monotonic_ns": ""},
            [], 7, 1, 1, water_accounting=accounting,
        )
        self.assertEqual(
            trial_payload["task_low_probability_cue_anticipatory_lick_trials_session"], 1
        )

    def test_behavior_counters_reject_unknown_protocol_role(self):
        accounting = reward.TaskWaterTelemetryAccounting(3.0)
        with self.assertRaisesRegex(ValueError, "Unknown trial image role"):
            accounting.record_completed_behavior_trial(
                {"trial_index": 99, "image_role": "unknown"}, False
            )

    def test_behavior_counters_ignore_partial_and_post_reward_licks(self):
        accounting = reward.TaskWaterTelemetryAccounting(3.0)
        partial = {"trial_index": 10, "image_role": "rewarded_high_1"}
        partial_runtime = {
            "trial_completed": False,
            "stim_request_monotonic_ns": 100_000_000_000,
        }
        reward.publish_completed_trial_telemetry(
            None, partial, partial_runtime, [], 1, 1, 1,
            water_accounting=accounting,
        )
        post_reward_only = {"trial_index": 11, "image_role": "rewarded_high_1"}
        post_reward_only_lick = [
            {"event_type": "lick_onset", "monotonic_ns": 101_500_000_000}
        ]
        runtime = {
            "trial_completed": True,
            "stim_request_monotonic_ns": 100_000_000_000,
        }
        result = reward.publish_completed_trial_telemetry(
            None, post_reward_only, runtime, post_reward_only_lick,
            1, 1, 1, water_accounting=accounting,
        )
        self.assertFalse(result["reward_delivered"])
        summary = accounting.summary()
        self.assertEqual(summary["task_rewarded_high_cue_trials_completed_session"], 1)
        self.assertEqual(summary["task_rewarded_high_cue_anticipatory_lick_trials_session"], 0)

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

    def _run_trials_until_iti_interrupt(self, interrupt_on_trial_number):
        trials = []
        for index in range(4):
            trials.append({
                "trial_index": index,
                "trial_number": index + 1,
                "block_number": 1,
                "image_role": "role_%d" % index,
                "image_category": "conditioned",
                "image_id": index + 1,
                "image_filename": "img_%d.png" % index,
                "reward_eligible": False,
                "reward_scheduled": False,
                "reward_omission_scheduled": False,
                "suction_scheduled": False,
                "planned_iti_duration_sec": 4.0,
            })

        loaded_raws = {
            trial["image_filename"]: {
                "first": Path("first.raw"), "second": Path("second.raw")
            }
            for trial in trials
        }
        raw_paths = {
            trial["image_filename"]: {
                "first": Path("first.raw"), "second": Path("second.raw")
            }
            for trial in trials
        }
        display_count = {"value": 0}
        wait_count = {"value": 0}

        def fake_display(screen, raw_path):
            del screen, raw_path
            display_count["value"] += 1
            base_ns = display_count["value"] * 1_000_000_000
            return SimpleNamespace(
                start_time=0, mean_interframe=0, stddev_interframe=0
            ), {
                "request_utc_iso": "iso",
                "request_unix_sec": "1.0",
                "request_unix_ns": base_ns,
                "request_perf_counter_ns": base_ns,
                "return_unix_ns": base_ns + 1,
                "return_perf_counter_ns": base_ns + 1,
                "duration_sec": 0.001,
            }

        def interrupting_wait(*args, **kwargs):
            del args, kwargs
            wait_count["value"] += 1
            if wait_count["value"] == interrupt_on_trial_number:
                raise KeyboardInterrupt()

        caller_runtime = {}
        gpio_client = FakeGPIOClient()
        with tempfile.TemporaryDirectory(prefix="reward_interrupt_runtime_") as temp_dir:
            with mock.patch.object(
                reward.base,
                "display_raw_with_timing",
                side_effect=fake_display,
            ), mock.patch.object(
                reward,
                "wait_until",
                side_effect=interrupting_wait,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    reward.run_trials(
                        SimpleNamespace(),
                        trials,
                        loaded_raws,
                        Path(temp_dir) / "gray.loaded",
                        raw_paths,
                        Path(temp_dir) / "gray.raw",
                        gpio_client,
                        Path(temp_dir) / "events.csv",
                        [],
                        6,
                        0.5,
                        3.5,
                        0.05,
                        runtime_by_trial=caller_runtime,
                    )
        return trials, caller_runtime

    def test_interruption_preserves_completed_trials(self):
        trials, runtime_by_trial = self._run_trials_until_iti_interrupt(3)
        self.assertEqual(sorted(runtime_by_trial), [0, 1, 2])
        for trial_index in (0, 1):
            self.assertTrue(runtime_by_trial[trial_index]["trial_executed"])
            self.assertTrue(runtime_by_trial[trial_index]["stim_presented"])
            self.assertTrue(runtime_by_trial[trial_index]["trial_completed"])
        self.assertTrue(runtime_by_trial[2]["trial_executed"])
        self.assertTrue(runtime_by_trial[2]["stim_presented"])
        self.assertFalse(runtime_by_trial[2]["trial_completed"])
        self.assertNotIn(3, runtime_by_trial)
        self.assertEqual(len(trials), 4)

    def test_interruption_during_current_iti_preserves_partial_trial(self):
        trials, runtime_by_trial = self._run_trials_until_iti_interrupt(2)
        self.assertTrue(runtime_by_trial[0]["trial_completed"])
        current = runtime_by_trial[1]
        self.assertTrue(current["trial_executed"])
        self.assertTrue(current["stim_presented"])
        self.assertFalse(current["trial_completed"])
        self.assertNotIn(2, runtime_by_trial)
        self.assertEqual(trials[1]["trial_index"], 1)

    def test_partial_trial_summary_preserves_executed_command_ids(self):
        trial = {
            "trial_index": 0, "trial_number": 1, "block_number": 1,
            "image_role": "rewarded_high_1", "image_category": "conditioned",
            "image_id": 1, "image_filename": "img.png",
            "reward_eligible": True, "reward_scheduled": True,
            "reward_omission_scheduled": False, "suction_scheduled": True,
            "planned_iti_duration_sec": 4.0,
        }
        runtime = {
            "trial_executed": True,
            "stim_presented": True,
            "trial_completed": False,
            "reward_command_id": "reward_1",
            "suction_command_id": "suction_1",
            "stim_request_monotonic_ns": 100_000_000_000,
        }
        events = [
            {"event_type": "reward_command_received", "command_id": "reward_1"},
            {"event_type": "reward_valve_on", "command_id": "reward_1", "monotonic_ns": 101_000_000_000},
            {"event_type": "reward_valve_off", "command_id": "reward_1", "monotonic_ns": 101_100_000_000},
            {"event_type": "reward_complete", "command_id": "reward_1"},
            {"event_type": "suction_command_received", "command_id": "suction_1"},
            {"event_type": "suction_on", "command_id": "suction_1", "monotonic_ns": 103_400_000_000},
            {"event_type": "suction_off", "command_id": "suction_1", "monotonic_ns": 103_450_000_000},
            {"event_type": "suction_complete", "command_id": "suction_1"},
        ]
        rows = reward.build_trial_summary([trial], {0: runtime}, events)
        self.assertTrue(rows[0]["trial_executed"])
        self.assertTrue(rows[0]["stim_presented"])
        self.assertFalse(rows[0]["trial_completed"])
        self.assertEqual(rows[0]["reward_command_id"], "reward_1")
        self.assertEqual(rows[0]["suction_command_id"], "suction_1")
        self.assertEqual(rows[0]["suction_on_monotonic_ns"], 103_400_000_000)

    def test_partial_qc_does_not_mark_executed_commands_unexpected(self):
        def make_trial(index, reward_scheduled, suction_scheduled):
            return {
                "trial_index": index, "trial_number": index + 1,
                "block_number": 1, "image_role": "role_%d" % index,
                "image_category": "conditioned", "image_id": index + 1,
                "image_filename": "img_%d.png" % index,
                "reward_eligible": reward_scheduled,
                "reward_scheduled": reward_scheduled,
                "reward_omission_scheduled": False,
                "suction_scheduled": suction_scheduled,
                "planned_iti_duration_sec": 4.0,
            }

        executed_trial = make_trial(0, True, True)
        future_trial = make_trial(1, False, False)
        runtime = {
            "trial_executed": True,
            "stim_presented": True,
            "trial_completed": False,
            "reward_command_id": "reward_1",
            "suction_command_id": "suction_1",
            "stim_request_monotonic_ns": 100_000_000_000,
        }
        events = [
            {"event_type": "reward_command_received", "command_id": "reward_1"},
            {"event_type": "reward_valve_on", "command_id": "reward_1"},
            {"event_type": "reward_valve_off", "command_id": "reward_1"},
            {"event_type": "reward_complete", "command_id": "reward_1"},
            {"event_type": "suction_command_received", "command_id": "suction_1"},
            {"event_type": "suction_on", "command_id": "suction_1"},
            {"event_type": "suction_off", "command_id": "suction_1"},
            {"event_type": "suction_complete", "command_id": "suction_1"},
        ]
        rows = reward.build_trial_summary(
            [executed_trial, future_trial], {0: runtime}, events
        )
        qc = reward.build_session_qc(
            "interrupted-session",
            [executed_trial, future_trial],
            rows,
            events,
            reward_num_pulses=1,
        )
        self.assertFalse(qc["qc_pass"])
        self.assertEqual(qc["executed_trial_count"], 1)
        self.assertEqual(qc["completed_trial_count"], 0)
        self.assertEqual(qc["unexpected_reward_command_ids"], [])
        self.assertEqual(qc["unexpected_suction_command_ids"], [])

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
            caller_runtime = {}
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
                    runtime_by_trial=caller_runtime,
                )

        self.assertEqual(len(wait_calls), 1)
        self.assertIs(runtime_by_trial, caller_runtime)
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
        reward_tracker = reward.RewardVolumeTracker(10.0, 10.0, 1).preflight()

        def fake_wait(deadline, gpio_client, event_path, events, status_callback=None):
            calls.append("wait_with_status" if status_callback else "wait_without_status")
            if status_callback:
                with mock.patch.object(reward.time, "monotonic", return_value=deadline - 0.5):
                    status_callback(0.5)

        status_updates = []
        telemetry_calls = []
        telemetry_publisher = SimpleNamespace(
            publish_trial=lambda payload: (telemetry_calls.append("telemetry_trial"), calls.append("telemetry_trial"))
        )
        telemetry_state_reporter = SimpleNamespace(
            report_state=lambda payload, force=False: (telemetry_calls.append("telemetry_state"), calls.append("telemetry_state"))
        )
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
                reward_volume_tracker=reward_tracker,
                telemetry_publisher=telemetry_publisher,
                telemetry_state_reporter=telemetry_state_reporter,
                total_blocks=1,
                post_background_sec=5.0,
            )

        first_index = calls.index("segment1")
        reward_index = calls.index("reward")
        second_index = calls.index("segment2")
        self.assertEqual(
            calls[first_index:second_index + 1],
            ["segment1", "reward", "segment2"],
        )
        self.assertEqual(reward_index, first_index + 1)
        self.assertEqual(telemetry_calls[0], "telemetry_state")
        self.assertEqual(telemetry_calls.count("telemetry_trial"), 1)
        self.assertEqual(calls.count("wait_with_status"), 2)
        self.assertEqual(calls.count("status"), 2)
        self.assertLess(calls.index("wait_with_status"), calls.index("suction"))
        self.assertLess(calls.index("suction"), calls.index("verify_suction"))
        self.assertTrue(all(update[:2] == (1, 2) for update in status_updates))
        self.assertEqual(reward_tracker.delivered_reward_train_count, 1)

    def test_blocked_reward_transitions_to_gray_before_logging(self):
        trial = {
            "trial_index": 0, "trial_number": 1, "block_number": 1,
            "image_role": "rewarded_high_1", "image_category": "conditioned",
            "image_id": 1, "image_filename": "img.png",
            "reward_eligible": True, "reward_scheduled": True,
            "reward_omission_scheduled": False, "suction_scheduled": False,
            "planned_iti_duration_sec": 4.0,
        }
        first = Path("first.raw")
        second = Path("second.raw")
        gray = Path("gray.loaded")
        calls = []

        def fake_display(screen, raw_path):
            label = {first: "segment1", second: "segment2", gray: "safe_gray"}[raw_path]
            calls.append(label)
            base_ns = len(calls) * 1_000_000_000
            return SimpleNamespace(start_time=0, mean_interframe=0,
                                   stddev_interframe=0), {
                "request_utc_iso": "iso", "request_unix_sec": "1.0",
                "request_unix_ns": base_ns,
                "request_perf_counter_ns": base_ns,
                "return_unix_ns": base_ns + 1,
                "return_perf_counter_ns": base_ns + 1,
                "duration_sec": 0.001,
            }

        gpio = FakeGPIOClient()
        gpio.trigger_reward = mock.Mock(return_value="must_not_send")
        tracker = reward.RewardVolumeTracker(10.0, 10.0, 1).preflight()
        tracker.record_dispatched(manual=True)

        with mock.patch.object(reward.base, "display_raw_with_timing",
                               side_effect=fake_display), \
             mock.patch.object(reward, "append_event",
                               side_effect=lambda *args, **kwargs: calls.append("event")):
            with self.assertRaises(reward.RewardVolumeCapExceeded):
                reward.run_trials(
                    SimpleNamespace(), [trial],
                    {"img.png": {"first": first, "second": second}}, gray,
                    {"img.png": {"first": first, "second": second}},
                    Path("gray.raw"), gpio, Path("events.csv"), [],
                    6, 0.5, 3.5, 0.05,
                    reward_volume_tracker=tracker)

        self.assertEqual(calls[:2], ["segment1", "safe_gray"])
        self.assertNotIn("segment2", calls)
        self.assertGreaterEqual(calls.count("event"), 2)
        self.assertGreater(calls.index("event"), calls.index("safe_gray"))
        gpio.trigger_reward.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
