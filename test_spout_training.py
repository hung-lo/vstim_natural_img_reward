#!/usr/bin/env python3
"""Tests for the gray-screen spout-training protocol."""

import csv
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_spout_training as training


class SpoutTrainingTests(unittest.TestCase):
    def test_manual_bait_mode_is_default_and_legacy_modes_parse(self):
        self.assertEqual(training.parse_args(["--simulate-gpio"]).bait_mode, "manual")
        self.assertEqual(
            training.parse_args(["--simulate-gpio", "--bait-drops", "4"]).bait_mode,
            "auto",
        )
        self.assertEqual(
            training.parse_args(["--simulate-gpio", "--no-bait"]).bait_mode,
            "none",
        )
        self.assertEqual(training.parse_args(["--simulate-gpio"]).manual_start_delay_sec, 0.0)

    def test_terminal_key_reader_restores_terminal(self):
        stream = mock.Mock()
        stream.fileno.return_value = 7
        stream.isatty.return_value = True
        with mock.patch.object(training.termios, "tcgetattr", return_value=[1, 2]), \
                mock.patch.object(training.tty, "setcbreak"), \
                mock.patch.object(training.termios, "tcsetattr") as restore:
            reader = training.TerminalKeyReader(stream)
            with reader:
                pass
            restore.assert_called_once_with(7, training.termios.TCSADRAIN, [1, 2])

    def test_telemetry_cli_options(self):
        args = training.parse_args([
            "--simulate-gpio", "--telemetry-host", "monitor",
            "--telemetry-port", "6000", "--no-telemetry",
        ])
        self.assertEqual(args.telemetry_host, "monitor")
        self.assertEqual(args.telemetry_port, 6000)
        self.assertTrue(args.no_telemetry)

    def test_spout_telemetry_payloads_are_gray_screen_safe(self):
        state = {
            "phase": "REWARD", "maximum_training_rewards": 3,
            "criterion_window_rewards": 20, "criterion_success_fraction": 0.8,
            "reward_volume_ul": 5.0, "reward_to_suction_delay_sec": 2.5,
            "training_reward_index": 2, "training_pass_reward_index": None,
            "training_passed": False,
        }
        session = training.build_spout_session_payload("s1", "m1", state)
        trial = training.build_spout_trial_payload(
            "s1", "m1", {"training_reward_index": 2, "retrieval_success": True}, state)
        self.assertEqual(session["protocol_name"], "spout_training")
        self.assertEqual(session["training_reward_index"], 2)
        self.assertIsNone(session["training_pass_reward_index"])
        self.assertIsNone(session["image"])
        self.assertIsNone(session["image_role"])
        self.assertEqual(trial["message_type"], "trial_complete")
        self.assertTrue(trial["reward_delivered"])
        self.assertTrue(trial["reward_contacted"])
        self.assertIsNone(trial["image"])

    def test_spout_telemetry_payload_preserves_failed_retrieval(self):
        trial = training.build_spout_trial_payload(
            "s1", "m1",
            {"training_reward_index": 1, "retrieval_success": False},
            {"maximum_training_rewards": 3},
        )
        self.assertFalse(trial["reward_contacted"])
        self.assertFalse(trial["retrieval_success"])

    def test_spout_telemetry_payload_reports_authoritative_cumulative_water(self):
        payload = training.build_spout_state_payload(
            "s1", "m1", {
                "maximum_training_rewards": 3,
                "reward_volume_ul": 5.0,
                "completed_training_reward_count": 3,
                "retrieval_success_count_session": 2,
                "retrieval_failure_count_session": 1,
                "task_water_delivered_ul_session": 15.0,
                "bait_water_ul_session": 10.0,
                "total_water_ul_session": 25.0,
            },
        )
        self.assertEqual(payload["completed_training_reward_count"], 3)
        self.assertEqual(payload["task_water_delivered_ul_session"], 15.0)
        self.assertEqual(payload["bait_water_ul_session"], 10.0)
        self.assertEqual(payload["total_water_ul_session"], 25.0)

    def test_spout_telemetry_payload_reports_recent_criterion(self):
        payload = training.build_spout_trial_payload(
            "s1", "m1", {"training_reward_index": 20,
                         "retrieval_success": True,
                         "recent_20_success_count": 16,
                         "recent_20_success_fraction": 0.8,
                         "criterion_evaluable": True,
                         "training_passed": True},
            {"maximum_training_rewards": 60},
        )
        self.assertEqual(payload["recent_20_success_count"], 16)
        self.assertEqual(payload["recent_20_success_fraction"], 0.8)
        self.assertTrue(payload["criterion_evaluable"])
        self.assertTrue(payload["training_passed"])

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

    def test_final_qc_requires_normal_completion_and_bait_hardware(self):
        hardware_qc = {"qc_pass": True, "qc_fail_reasons": []}
        failed = training.finalize_training_qc(
            hardware_qc, False, 1, 0, 1, 1,
        )
        self.assertFalse(failed["qc_pass"])
        self.assertFalse(failed["bait_hardware_complete"])
        self.assertIn("bait hardware incomplete", failed["qc_fail_reasons"])
        self.assertIn("session did not complete normally", failed["qc_fail_reasons"])

        normal_behavior_failure = training.finalize_training_qc(
            {"qc_pass": True, "qc_fail_reasons": []}, True, 0, 0, 0, 0,
        )
        self.assertTrue(normal_behavior_failure["qc_pass"])

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

    def _run_fake_session(self, mode="normal", bait=True, max_rewards=1,
                          telemetry_failure=False):
        class Clock(object):
            def __init__(self):
                self.now = 100.0

            def monotonic(self):
                return self.now

            def monotonic_ns(self):
                return int(self.now * 1_000_000_000)

            def time_ns(self):
                return int((1_700_000_000.0 + self.now) * 1_000_000_000)

            def sleep(self, seconds):
                self.now += max(0.001, float(seconds))

        class FakeGPIO(object):
            def __init__(self, clock, failure_mode):
                self.clock = clock
                self.failure_mode = failure_mode
                self.events = []
                self.contexts = []
                self.reward_count = 0
                self.suction_count = 0
                self.suction_calls = []

            def start(self):
                return {"worker_pid": 1, "simulate_gpio": True}

            def set_context(self, context):
                self.contexts.append(dict(context))

            def _event(self, event_type, command_id, phase, offset=0.0):
                return {
                    "message_type": "event", "event_type": event_type,
                    "command_id": command_id, "phase": phase,
                    "monotonic_ns": int((self.clock.now + offset) * 1e9),
                    "unix_time_ns": int((1_700_000_000.0 + self.clock.now + offset) * 1e9),
                }

            def trigger_reward(self, context):
                self.reward_count += 1
                command_id = "reward_%d" % self.reward_count
                phase = context.get("phase", "")
                self.events.extend([
                    self._event("reward_command_received", command_id, phase),
                    self._event("reward_valve_on", command_id, phase, 0.1),
                    self._event("reward_valve_off", command_id, phase, 0.2),
                ])
                omit = (
                    (self.failure_mode == "bait_reward" and phase == "spout_training_bait")
                    or (self.failure_mode == "training_reward" and phase == "spout_training")
                )
                if not omit:
                    self.events.append(self._event("reward_complete", command_id, phase, 0.3))
                return command_id

            def trigger_suction(self, context):
                self.suction_count += 1
                command_id = "suction_%d" % self.suction_count
                phase = context.get("phase", "")
                self.suction_calls.append((self.clock.now, phase))
                self.events.extend([
                    self._event("suction_command_received", command_id, phase),
                    self._event("suction_on", command_id, phase, 0.01),
                    self._event("suction_off", command_id, phase, 0.02),
                ])
                omit = (
                    (self.failure_mode == "bait_suction" and phase == "spout_training_bait")
                    or (self.failure_mode == "training_suction" and phase == "spout_training")
                )
                if not omit:
                    self.events.append(self._event("suction_complete", command_id, phase, 0.03))
                return command_id

            def drain_events(self):
                events, self.events = self.events, []
                return events

            def is_alive(self):
                return True

            def shutdown(self):
                return [self._event("gpio_worker_stopped", "shutdown-final", "shutdown")]

        with tempfile.TemporaryDirectory(prefix="spout_integration_") as directory:
            root = Path(directory)
            config_path = root / "hardware.json"
            config_path.write_text(json.dumps({
                "reward_pin_bcm": 19, "suction_pin_bcm": 25,
                "lick_pin_bcm": 26, "suction_duration_sec": 0.1,
            }))
            args = SimpleNamespace(
                hardware_config=str(config_path), mouse_id="test_mouse",
                output_root=str(root), simulate_gpio=True,
                max_rewards=max_rewards, interval_min_sec=0.01,
                interval_max_sec=0.01, settle_sec=0.0,
                criterion_window=20, criterion_fraction=0.8,
                no_bait=not bait, bait_drops=1 if bait else 0,
                telemetry_host="127.0.0.1", telemetry_port=5055,
                no_telemetry=not telemetry_failure,
            )
            clock = Clock()
            client = FakeGPIO(clock, mode)
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    training, "BehaviorGPIOClient", return_value=client))
                stack.enter_context(mock.patch.object(training, "time", clock))
                if telemetry_failure:
                    publisher = mock.Mock()
                    publisher.enabled = True
                    publisher.start.return_value = True
                    publisher.publish_session.side_effect = RuntimeError("telemetry down")
                    publisher.publish_state.side_effect = RuntimeError("telemetry down")
                    publisher.publish_trial.side_effect = RuntimeError("telemetry down")
                    stack.enter_context(mock.patch.object(
                        training, "TelemetryPublisher", return_value=publisher))
                error = None
                try:
                    result = training.run_training(args)
                except Exception as exc:
                    result = None
                    error = exc
            session_root = next(root.glob("test_mouse_*_spout_training"))
            metadata_path = next(session_root.glob("*_metadata.json"))
            qc_path = next(session_root.glob("*_session_qc.json"))
            event_path = next(session_root.glob("*_event_log.csv"))
            summary_path = next(session_root.glob("*_reward_summary.csv"))
            metadata = json.loads(metadata_path.read_text())
            qc = json.loads(qc_path.read_text())
            artifacts = {
                "metadata_exists": metadata_path.exists(),
                "qc_exists": qc_path.exists(),
                "event_log": event_path.read_text(),
                "reward_summary": summary_path.read_text(),
            }
            return result, error, client, metadata, qc, artifacts

    def test_integration_telemetry_failure_does_not_change_training_outputs(self):
        result, error, client, metadata, qc, artifacts = self._run_fake_session(
            bait=False, telemetry_failure=True)
        self.assertIsNone(error)
        self.assertIsNotNone(result)
        self.assertEqual(client.reward_count, 1)
        self.assertTrue(metadata["session_completed"])
        self.assertTrue(qc["qc_pass"])
        self.assertIn("reward_summary", artifacts)

    def test_integration_bait_reward_failure_finalizes_and_does_not_train(self):
        _, error, client, metadata, qc, artifacts = self._run_fake_session("bait_reward")
        self.assertIn("Bait reward", str(error))
        self.assertEqual(client.reward_count, 1)
        self.assertEqual(metadata["completed_bait_reward_count"], 0)
        self.assertFalse(metadata["session_completed"])
        self.assertFalse(metadata["qc_pass"])
        self.assertTrue(artifacts["metadata_exists"])
        self.assertTrue(artifacts["qc_exists"])
        self.assertTrue(qc["qc_fail_reasons"])

    def test_integration_bait_suction_failure_finalizes_and_does_not_train(self):
        _, error, client, metadata, qc, _ = self._run_fake_session("bait_suction")
        self.assertIn("Bait suction", str(error))
        self.assertEqual(client.reward_count, 1)
        self.assertEqual(metadata["completed_bait_reward_count"], 1)
        self.assertFalse(metadata["bait_hardware_complete"])
        self.assertFalse(qc["qc_pass"])

    def test_integration_training_suction_failure_accounts_completed_reward(self):
        _, error, client, metadata, qc, _ = self._run_fake_session("training_suction", bait=False)
        self.assertIn("Incomplete reward/suction", str(error))
        self.assertEqual(client.reward_count, 1)
        self.assertEqual(metadata["attempted_training_reward_count"], 1)
        self.assertEqual(metadata["completed_training_reward_count"], 1)
        self.assertEqual(metadata["actual_training_water_ul"], 5.0)
        self.assertFalse(qc["qc_pass"])

    def test_integration_training_reward_failure_does_not_account_water(self):
        _, error, client, metadata, qc, _ = self._run_fake_session("training_reward", bait=False)
        self.assertIn("Incomplete reward/suction", str(error))
        self.assertEqual(client.reward_count, 1)
        self.assertEqual(metadata["completed_training_reward_count"], 0)
        self.assertEqual(metadata["actual_training_water_ul"], 0.0)
        self.assertFalse(qc["qc_pass"])

    def test_integration_normal_behavioral_failure_still_passes_qc(self):
        result, error, client, metadata, qc, _ = self._run_fake_session(bait=False)
        self.assertIsNone(error)
        self.assertIsNotNone(result)
        self.assertEqual(client.reward_count, 1)
        self.assertTrue(metadata["session_completed"])
        self.assertFalse(metadata["training_passed"])
        self.assertTrue(metadata["qc_pass"])
        self.assertTrue(qc["qc_pass"])

    def test_integration_shutdown_event_is_retained_once_and_context_progresses(self):
        _, error, client, metadata, _, artifacts = self._run_fake_session(bait=False)
        self.assertIsNone(error)
        event_log = artifacts["event_log"]
        self.assertEqual(event_log.count("shutdown-final"), 1)
        self.assertIn("spout_training_inter_reward", [context["phase"] for context in client.contexts])
        self.assertTrue(metadata["session_completed"])

    def test_integration_bait_suction_uses_actual_reward_on_timestamp(self):
        _, error, client, _, _, _ = self._run_fake_session(bait=True, max_rewards=1)
        self.assertIsNone(error)
        bait_suction_time = client.suction_calls[0][0]
        bait_reward_on_time = 100.1
        self.assertAlmostEqual(bait_suction_time - bait_reward_on_time, 2.5, places=1)

    def test_integration_delayed_episode_reanchors_effective_target(self):
        _, error, client, _, _, artifacts = self._run_fake_session(bait=False, max_rewards=2)
        self.assertIsNone(error)
        self.assertIn(
            "spout_training_inter_reward",
            [context["phase"] for context in client.contexts],
        )
        rows = list(csv.DictReader(artifacts["reward_summary"].splitlines()))
        self.assertEqual(len(rows), 2)
        self.assertGreaterEqual(
            int(rows[1]["effective_reward_target_monotonic_ns"]),
            int(rows[0]["reward_on_monotonic_ns"]) + 10_000_000,
        )
        self.assertAlmostEqual(
            float(rows[1]["software_reward_timing_error_sec"]),
            (int(rows[1]["reward_on_monotonic_ns"]) - int(rows[1]["effective_reward_target_monotonic_ns"])) / 1e9,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
