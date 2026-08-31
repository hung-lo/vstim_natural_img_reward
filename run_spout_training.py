#!/usr/bin/env python3
"""Gray-screen, open-loop lick-spout training session."""

from __future__ import print_function

import argparse
import csv
import json
import math
import random
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from reward_conditioning_gpio import BehaviorGPIOClient
from rig_telemetry import (
    DEFAULT_HOST as DEFAULT_TELEMETRY_HOST,
    DEFAULT_PORT as DEFAULT_TELEMETRY_PORT,
    TelemetryPublisher,
)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path("/mnt/hd")
DEFAULT_CONFIG = PROJECT_ROOT / "spout_training_config.json"
DEFAULT_REWARD_VOLUME_UL = 5.0
DEFAULT_PULSE_ON_SEC = 0.208
DEFAULT_PULSE_OFF_SEC = 0.01
DEFAULT_REWARD_PULSES = 1
DEFAULT_SUCTION_DELAY_SEC = 2.5
DEFAULT_SETTLE_SEC = 30.0
DEFAULT_INTERVAL_MIN_SEC = 8.0
DEFAULT_INTERVAL_MAX_SEC = 12.0
DEFAULT_MAX_REWARDS = 60
DEFAULT_CRITERION_WINDOW = 20
DEFAULT_CRITERION_MIN_REWARDS = 20
DEFAULT_CRITERION_FRACTION = 0.80
DEFAULT_BAIT_DROPS = 2
BAIT_INTERVAL_SEC = 5.0
EPISODE_POLL_SEC = 0.02
TELEMETRY_HEARTBEAT_SEC = 1.0

REWARD_EVENT_TYPES = (
    "reward_command_received", "reward_valve_on", "reward_valve_off",
    "reward_complete",
)
SUCTION_EVENT_TYPES = (
    "suction_command_received", "suction_on", "suction_off",
    "suction_complete",
)
REWARD_SUMMARY_FIELDS = [
    "training_reward_index", "planned_interval_sec",
    "planned_reward_target_monotonic_ns", "reward_command_id",
    "reward_command_unix_ns", "reward_command_monotonic_ns",
    "reward_on_unix_ns", "reward_on_monotonic_ns",
    "reward_off_unix_ns", "reward_off_monotonic_ns",
    "reward_complete_unix_ns", "reward_complete_monotonic_ns",
    "effective_reward_target_monotonic_ns", "software_reward_timing_error_sec", "suction_command_id",
    "suction_target_monotonic_ns", "suction_on_unix_ns",
    "suction_on_monotonic_ns", "suction_off_unix_ns",
    "suction_off_monotonic_ns", "suction_complete_unix_ns",
    "suction_complete_monotonic_ns", "software_suction_delay_sec",
    "lick_count_pre_reward_1p0_sec", "lick_count_reward_to_0p5_sec",
    "lick_count_reward_to_1p0_sec", "lick_count_reward_to_2p5_sec",
    "first_lick_latency_sec", "retrieval_success", "recent_20_success_count",
    "recent_20_success_fraction", "criterion_evaluable",
    "criterion_passed_after_this_reward",
]
LICK_FIELDS = [
    "unix_time_ns", "monotonic_ns", "event_type", "phase",
    "training_reward_index", "reward_command_id",
]
EVENT_FIELDS = [
    "utc_iso", "unix_time_utc_sec", "unix_time_ns", "monotonic_ns",
    "event_type", "message_type", "phase", "training_reward_index",
    "reward_command_id", "suction_command_id", "command_id", "lick_edge",
    "notes",
]


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_commit():
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL, universal_newlines=True,
        ).strip()
    except Exception:
        return ""


def git_dirty():
    import subprocess
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL, universal_newlines=True,
        ).strip())
    except Exception:
        return False


def prepare_gray_display(output_root):
    """Initialize one static gray frame; no experimental image is loaded."""
    import run_stringer_vstim as base
    import rpg
    display_dir = Path(output_root) / "display_cache"
    display_dir.mkdir(parents=True, exist_ok=True)
    gray_path = display_dir / "spout_training_gray.raw"
    canvas = base.build_canvas(None, base.SCREEN_RESOLUTION, photodiode_on=False)
    base.convert_canvas_to_rpg_raw(
        rpg, canvas, gray_path, 1.0 / float(base.REFRESH_RATE_HZ),
    )
    screen = rpg.Screen(
        base.SCREEN_RESOLUTION[0], base.SCREEN_RESOLUTION[1],
        base.SCREEN_COLORMODE, base.REFRESH_RATE_HZ,
    )
    screen.display_raw(gray_path)
    return screen


def write_rows(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_training_intervals(max_rewards=DEFAULT_MAX_REWARDS,
                             interval_min_sec=DEFAULT_INTERVAL_MIN_SEC,
                             interval_max_sec=DEFAULT_INTERVAL_MAX_SEC,
                             seed=None):
    """Generate the full random interval sequence before training starts."""
    max_rewards = int(max_rewards)
    low, high = float(interval_min_sec), float(interval_max_sec)
    if max_rewards < 1:
        raise ValueError("max_rewards must be at least 1")
    if low <= 0 or high < low:
        raise ValueError("Require 0 < interval_min_sec <= interval_max_sec")
    rng = random.Random(seed)
    return [0.0] + [rng.uniform(low, high) for _ in range(max_rewards - 1)]


def anchor_training_schedule(start_monotonic_ns, intervals):
    """Anchor absolute targets after baiting and settling have completed."""
    target_ns = int(start_monotonic_ns)
    rows = []
    for index, interval in enumerate(intervals, start=1):
        if index > 1:
            target_ns += int(round(interval * 1_000_000_000.0))
        rows.append({
            "training_reward_index": index,
            "planned_interval_sec": interval,
            "planned_reward_target_monotonic_ns": target_ns,
        })
    return rows


def build_training_schedule(start_monotonic_ns, max_rewards=DEFAULT_MAX_REWARDS,
                            interval_min_sec=DEFAULT_INTERVAL_MIN_SEC,
                            interval_max_sec=DEFAULT_INTERVAL_MAX_SEC,
                            seed=None):
    """Backward-compatible convenience wrapper for an already-known anchor."""
    return anchor_training_schedule(
        start_monotonic_ns,
        build_training_intervals(max_rewards, interval_min_sec, interval_max_sec, seed),
    )


def evaluate_training_criterion(successes, minimum_rewards=DEFAULT_CRITERION_MIN_REWARDS,
                                window=DEFAULT_CRITERION_WINDOW,
                                fraction=DEFAULT_CRITERION_FRACTION):
    """Evaluate only the most recent completed scheduled reward episodes."""
    values = [bool(value) for value in successes]
    minimum_rewards, window = int(minimum_rewards), int(window)
    if minimum_rewards < 1 or window < 1 or not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("Invalid training criterion configuration")
    recent = values[-window:]
    evaluable = len(values) >= minimum_rewards and len(recent) >= window
    count = sum(recent)
    result_fraction = count / float(len(recent)) if recent else None
    return {
        "recent_success_count": count,
        "recent_success_fraction": result_fraction,
        "criterion_evaluable": evaluable,
        "criterion_passed": bool(evaluable and result_fraction >= float(fraction)),
    }


def _event_ns(event):
    try:
        return int(event.get("monotonic_ns"))
    except (TypeError, ValueError):
        return None


def first_command_event(events, event_type, command_id):
    for event in events:
        if (event.get("event_type") == event_type
                and event.get("command_id") == command_id):
            return event
    return None


def completed_command_ids(events, command_ids, complete_event_type):
    """Return unique command IDs with exactly one completion event."""
    return {
        command_id for command_id in set(command_ids)
        if sum(
            1 for event in events
            if event.get("command_id") == command_id
            and event.get("event_type") == complete_event_type
        ) == 1
    }


def _event_timestamp_fields(event):
    if event is None:
        return {}
    return {
        "unix_ns": event.get("unix_time_ns", ""),
        "monotonic_ns": event.get("monotonic_ns", ""),
    }


def _events_between(events, event_type, lower_ns, upper_ns):
    return [
        event for event in events
        if event.get("event_type") == event_type
        and _event_ns(event) is not None
        and lower_ns <= _event_ns(event) < upper_ns
    ]


def compute_reward_lick_metrics(events, reward_on_monotonic_ns, suction_on_monotonic_ns):
    """Compute retrieval from actual GPIO timestamps, not context labels."""
    reward_on_ns = int(reward_on_monotonic_ns)
    suction_on_ns = int(suction_on_monotonic_ns)
    lick_events = [event for event in events if event.get("event_type") == "lick_onset"]
    pre = _events_between(lick_events, "lick_onset", reward_on_ns - 1_000_000_000, reward_on_ns)
    reward_to_half = _events_between(lick_events, "lick_onset", reward_on_ns, reward_on_ns + 500_000_000)
    reward_to_one = _events_between(lick_events, "lick_onset", reward_on_ns, reward_on_ns + 1_000_000_000)
    reward_to_suction = _events_between(lick_events, "lick_onset", reward_on_ns, suction_on_ns)
    first_latency = None
    if reward_to_suction:
        first_latency = (_event_ns(reward_to_suction[0]) - reward_on_ns) / 1e9
    return {
        "lick_count_pre_reward_1p0_sec": len(pre),
        "lick_count_reward_to_0p5_sec": len(reward_to_half),
        "lick_count_reward_to_1p0_sec": len(reward_to_one),
        "lick_count_reward_to_2p5_sec": len(reward_to_suction),
        "first_lick_latency_sec": first_latency,
        "retrieval_success": bool(reward_to_suction),
    }


def build_training_qc(reward_rows, events, training_passed, criterion,
                      attempted_reward_command_ids=None,
                      attempted_suction_command_ids=None):
    """Return system QC separately from the mouse's behavioral outcome."""
    reward_ids = set(attempted_reward_command_ids or {
        row.get("reward_command_id") for row in reward_rows
        if row.get("reward_command_id")
    })
    suction_ids = set(attempted_suction_command_ids or {
        row.get("suction_command_id") for row in reward_rows
        if row.get("suction_command_id")
    })
    def count(types, ids):
        return sum(1 for event in events if event.get("command_id") in ids and event.get("event_type") in types)
    expected_pulses = len(reward_ids)
    expected_suctions = len(suction_ids)
    observed_reward_ids = {
        event.get("command_id") for event in events
        if event.get("event_type") in REWARD_EVENT_TYPES and event.get("command_id")
        and event.get("phase") != "spout_training_bait"
    }
    observed_suction_ids = {
        event.get("command_id") for event in events
        if event.get("event_type") in SUCTION_EVENT_TYPES and event.get("command_id")
        and event.get("phase") != "spout_training_bait"
    }
    reward_received_counts = {
        command_id: count({"reward_command_received"}, {command_id})
        for command_id in reward_ids
    }
    suction_received_counts = {
        command_id: count({"suction_command_received"}, {command_id})
        for command_id in suction_ids
    }
    checks = {
        "planned_scheduled_training_reward_count": expected_pulses,
        "attempted_training_reward_count": expected_pulses,
        "reward_command_received_count": count({"reward_command_received"}, reward_ids),
        "reward_complete_count": count({"reward_complete"}, reward_ids),
        "reward_valve_on_count": count({"reward_valve_on"}, reward_ids),
        "reward_valve_off_count": count({"reward_valve_off"}, reward_ids),
        "planned_suction_count": expected_suctions,
        "suction_command_received_count": count({"suction_command_received"}, suction_ids),
        "suction_on_count": count({"suction_on"}, suction_ids),
        "suction_off_count": count({"suction_off"}, suction_ids),
        "suction_complete_count": count({"suction_complete"}, suction_ids),
        "missing_reward_command_ids": sorted(reward_ids - observed_reward_ids),
        "unexpected_reward_command_ids": sorted(observed_reward_ids - reward_ids),
        "duplicate_reward_command_ids": sorted(
            command_id for command_id, value in reward_received_counts.items() if value > 1
        ),
        "incomplete_reward_command_ids": sorted(
            command_id for command_id in reward_ids
            if count({"reward_complete"}, {command_id}) != 1
        ),
        "missing_suction_command_ids": sorted(suction_ids - observed_suction_ids),
        "unexpected_suction_command_ids": sorted(observed_suction_ids - suction_ids),
        "duplicate_suction_command_ids": sorted(
            command_id for command_id, value in suction_received_counts.items() if value > 1
        ),
        "incomplete_suction_command_ids": sorted(
            command_id for command_id in suction_ids
            if count({"suction_complete"}, {command_id}) != 1
        ),
        "training_passed": bool(training_passed),
        "criterion_window_size": int(criterion.get("window", DEFAULT_CRITERION_WINDOW)),
        "final_criterion_success_fraction": criterion.get("recent_success_fraction"),
    }
    checks["qc_pass"] = (
        checks["reward_command_received_count"] == expected_pulses
        and checks["reward_complete_count"] == expected_pulses
        and checks["reward_valve_on_count"] == expected_pulses
        and checks["reward_valve_off_count"] == expected_pulses
        and checks["suction_command_received_count"] == expected_suctions
        and checks["suction_on_count"] == expected_suctions
        and checks["suction_off_count"] == expected_suctions
        and checks["suction_complete_count"] == expected_suctions
        and not checks["missing_reward_command_ids"]
        and not checks["unexpected_reward_command_ids"]
        and not checks["duplicate_reward_command_ids"]
        and not checks["incomplete_reward_command_ids"]
        and not checks["missing_suction_command_ids"]
        and not checks["unexpected_suction_command_ids"]
        and not checks["duplicate_suction_command_ids"]
        and not checks["incomplete_suction_command_ids"]
    )
    checks["qc_fail_reasons"] = []
    for field, expected in (
        ("reward_command_received_count", expected_pulses),
        ("reward_complete_count", expected_pulses),
        ("reward_valve_on_count", expected_pulses),
        ("reward_valve_off_count", expected_pulses),
        ("suction_command_received_count", expected_suctions),
        ("suction_on_count", expected_suctions),
        ("suction_off_count", expected_suctions),
        ("suction_complete_count", expected_suctions),
    ):
        if checks[field] != expected:
            checks["qc_fail_reasons"].append(
                "%s %s != expected %s" % (field, checks[field], expected)
            )
    for field in (
        "missing_reward_command_ids", "unexpected_reward_command_ids",
        "duplicate_reward_command_ids", "incomplete_reward_command_ids",
        "missing_suction_command_ids", "unexpected_suction_command_ids",
        "duplicate_suction_command_ids", "incomplete_suction_command_ids",
    ):
        if checks[field]:
            checks["qc_fail_reasons"].append("%s: %s" % (field, checks[field]))
    return checks


def finalize_training_qc(qc, session_completed,
                         attempted_bait_reward_count,
                         completed_bait_reward_count,
                         attempted_bait_suction_count,
                         completed_bait_suction_count):
    """Apply session-level completion and bait hardware requirements."""
    bait_hardware_complete = (
        int(attempted_bait_reward_count) == int(completed_bait_reward_count)
        and int(attempted_bait_suction_count) == int(completed_bait_suction_count)
    )
    bait_qc_required = (
        int(attempted_bait_reward_count) > 0
        or int(attempted_bait_suction_count) > 0
    )
    bait_qc_pass = not bait_qc_required or bait_hardware_complete
    qc["training_hardware_qc_pass"] = bool(qc.get("qc_pass", False))
    qc["bait_hardware_complete"] = bait_hardware_complete
    qc["bait_qc_pass"] = bait_qc_pass
    qc["session_completed"] = bool(session_completed)
    if not bait_qc_pass and "bait hardware incomplete" not in qc["qc_fail_reasons"]:
        qc["qc_fail_reasons"].append("bait hardware incomplete")
    if not session_completed and "session did not complete normally" not in qc["qc_fail_reasons"]:
        qc["qc_fail_reasons"].append("session did not complete normally")
    qc["qc_pass"] = bool(
        qc["training_hardware_qc_pass"]
        and bait_qc_pass
        and session_completed
    )
    return qc


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mouse-id", default="mouse")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--simulate-gpio", action="store_true")
    parser.add_argument("--max-rewards", type=int, default=DEFAULT_MAX_REWARDS)
    parser.add_argument("--interval-min-sec", type=float, default=DEFAULT_INTERVAL_MIN_SEC)
    parser.add_argument("--interval-max-sec", type=float, default=DEFAULT_INTERVAL_MAX_SEC)
    parser.add_argument("--settle-sec", type=float, default=DEFAULT_SETTLE_SEC)
    parser.add_argument("--criterion-window", type=int, default=DEFAULT_CRITERION_WINDOW)
    parser.add_argument("--criterion-fraction", type=float, default=DEFAULT_CRITERION_FRACTION)
    parser.add_argument("--telemetry-host", default=DEFAULT_TELEMETRY_HOST)
    parser.add_argument("--telemetry-port", type=int, default=DEFAULT_TELEMETRY_PORT)
    parser.add_argument("--no-telemetry", action="store_true")
    parser.add_argument("--no-bait", action="store_true")
    parser.add_argument("--bait-drops", type=int, default=DEFAULT_BAIT_DROPS)
    args = parser.parse_args(argv)
    if not args.simulate_gpio and not Path(args.hardware_config).exists():
        parser.error("hardware config does not exist: %s" % args.hardware_config)
    if args.max_rewards < 1 or args.bait_drops < 0:
        parser.error("reward counts must be nonnegative, with max-rewards at least 1")
    return args


def load_config(path, simulate_gpio=False):
    config = {
        "reward_pin_bcm": 19,
        "suction_pin_bcm": 25,
        "lick_pin_bcm": 26,
        "suction_duration_sec": 0.1,
        "lick_bounce_time_sec": None,
    }
    if Path(path).exists():
        with Path(path).open() as handle:
            config.update(json.load(handle))
    training = config.get("spout_training", {})
    config.update(training if isinstance(training, dict) else {})
    config.update({
        "reward_num_pulses": int(config.get("reward_num_pulses", DEFAULT_REWARD_PULSES)),
        "reward_pulse_on_sec": float(config.get("reward_pulse_on_sec", DEFAULT_PULSE_ON_SEC)),
        "reward_pulse_off_sec": float(config.get("reward_pulse_off_sec", DEFAULT_PULSE_OFF_SEC)),
        "reward_volume_ul": float(config.get("reward_volume_ul", DEFAULT_REWARD_VOLUME_UL)),
        "suction_delay_from_stim_onset_sec": DEFAULT_SUCTION_DELAY_SEC,
        "simulate_gpio": bool(simulate_gpio),
    })
    if config["reward_num_pulses"] != 1:
        raise RuntimeError("Spout training requires reward_num_pulses=1.")
    if config["reward_pulse_on_sec"] <= 0 or config["reward_volume_ul"] <= 0:
        raise RuntimeError("Reward pulse and declared volume must be positive.")
    return config


def _timestamp_fields():
    unix_ns = time.time_ns()
    return {
        "unix_time_ns": unix_ns,
        "monotonic_ns": time.monotonic_ns(),
        "utc_iso": datetime.fromtimestamp(unix_ns / 1e9, timezone.utc).isoformat(),
        "unix_time_utc_sec": "%.9f" % (unix_ns / 1e9),
    }


def _context(phase, index="", reward_command_id="", suction_command_id=""):
    return {
        "phase": phase, "training_reward_index": index,
        "reward_command_id": reward_command_id,
        "suction_command_id": suction_command_id,
    }


def _spout_cumulative_fields(state):
    return {
        "training_reward_index": state.get("training_reward_index"),
        "attempted_training_reward_count": state.get("attempted_training_reward_count", 0),
        "completed_training_reward_count": state.get("completed_training_reward_count", 0),
        "retrieval_success_count_session": state.get("retrieval_success_count_session", 0),
        "retrieval_failure_count_session": state.get("retrieval_failure_count_session", 0),
        "recent_20_success_count": state.get("recent_20_success_count"),
        "recent_20_success_fraction": state.get("recent_20_success_fraction"),
        "criterion_evaluable": bool(state.get("criterion_evaluable", False)),
        "training_passed": bool(state.get("training_passed", False)),
        "reward_volume_ul": state.get("reward_volume_ul"),
        "task_water_delivered_ul_session": state.get("task_water_delivered_ul_session", 0.0),
        "bait_water_ul_session": state.get("bait_water_ul_session", 0.0),
        "total_water_ul_session": state.get("total_water_ul_session", 0.0),
        "total_lick_onset_count_session": state.get("total_lick_onset_count_session", 0),
    }


def build_spout_session_payload(session_id, mouse_id, state):
    payload = {
        "protocol_name": "spout_training",
        "session_id": session_id,
        "mouse_id": mouse_id,
        "phase": state.get("phase", "STARTING"),
        "maximum_training_rewards": state.get("maximum_training_rewards"),
        "criterion_window_rewards": state.get("criterion_window_rewards"),
        "criterion_success_fraction": state.get("criterion_success_fraction"),
        "reward_volume_ul": state.get("reward_volume_ul"),
        "reward_to_suction_delay_sec": state.get("reward_to_suction_delay_sec"),
        "requested_bait_reward_count": state.get("requested_bait_reward_count", 0),
        "planned_training_water_ul": state.get("planned_training_water_ul"),
        "telemetry_enabled": bool(state.get("telemetry_enabled", False)),
        "image": None,
        "image_role": None,
    }
    payload.update(_spout_cumulative_fields(state))
    return payload


def build_spout_state_payload(session_id, mouse_id, state):
    payload = build_spout_session_payload(session_id, mouse_id, state)
    payload["message_type"] = "state"
    payload["next_reward_in_sec"] = state.get("next_reward_in_sec")
    payload["planned_remaining_eta_sec"] = state.get("planned_remaining_eta_sec")
    payload["bait_index"] = state.get("bait_index")
    payload["bait_total"] = state.get("bait_total")
    payload["completed_bait_reward_count"] = state.get("completed_bait_reward_count", 0)
    payload["completed_bait_suction_count"] = state.get("completed_bait_suction_count", 0)
    payload["failure_summary"] = state.get("failure_summary", "")
    return payload


def build_spout_trial_payload(session_id, mouse_id, row, state):
    payload = build_spout_state_payload(session_id, mouse_id, state)
    payload.update({
        "message_type": "trial_complete",
        "training_reward_index": row.get("training_reward_index"),
        "maximum_training_rewards": state.get("maximum_training_rewards"),
        "trial": row.get("training_reward_index"),
        "trial_total": state.get("maximum_training_rewards"),
        "reward_delivered": True,
        "reward_contacted": bool(row.get("retrieval_success")),
        "retrieval_success": bool(row.get("retrieval_success")),
    })
    for field in (
        "first_lick_latency_sec", "lick_count_pre_reward_1p0_sec",
        "lick_count_reward_to_0p5_sec", "lick_count_reward_to_1p0_sec",
        "lick_count_reward_to_2p5_sec", "recent_20_success_count",
        "recent_20_success_fraction", "criterion_evaluable", "training_passed",
    ):
        payload[field] = row.get(field, state.get(field))
    payload.update({"block": None, "block_total": None, "image": None, "image_role": None})
    return payload


def run_training(args):
    config = load_config(args.hardware_config, args.simulate_gpio)
    session_id = "%s_%s_spout_training" % (args.mouse_id, utc_stamp())
    root = Path(args.output_root) / session_id
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / (session_id + "_metadata.json")
    event_path = root / (session_id + "_event_log.csv")
    summary_path = root / (session_id + "_reward_summary.csv")
    lick_path = root / (session_id + "_lick_events.csv")
    qc_path = root / (session_id + "_session_qc.json")
    planned_path = root / (session_id + "_planned_rewards.csv")
    seed = random.SystemRandom().randrange(0, 2 ** 63)
    interval_plan = build_training_intervals(
        args.max_rewards, args.interval_min_sec, args.interval_max_sec, seed,
    )
    schedule = []
    all_events, reward_rows, successes = [], [], []
    requested_bait_reward_count = 0 if args.no_bait else int(args.bait_drops)
    attempted_bait_reward_count = 0
    completed_bait_reward_count = 0
    completed_bait_suction_count = 0
    attempted_bait_reward_command_ids = []
    attempted_bait_suction_command_ids = []
    attempted_training_reward_command_ids = []
    attempted_training_suction_command_ids = []
    attempted_training_reward_count = 0
    interrupted = False
    training_passed = False
    pass_index = None
    old_handler = signal.getsignal(signal.SIGINT)
    metadata = {}
    failure_exc = None
    failure_summary = ""
    seen_event_keys = set()

    def write_event(event):
        event_key = (
            event.get("event_type"), event.get("command_id"),
            event.get("monotonic_ns"), event.get("unix_time_ns"),
        )
        if event_key in seen_event_keys:
            return
        seen_event_keys.add(event_key)
        all_events.append(dict(event))
        with event_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS, extrasaction="ignore")
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow({field: event.get(field, "") for field in EVENT_FIELDS})
        if event.get("event_type") in ("lick_onset", "lick_offset"):
            with lick_path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=LICK_FIELDS, extrasaction="ignore")
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow({field: event.get(field, "") for field in LICK_FIELDS})

    client = BehaviorGPIOClient(config)
    screen = None
    telemetry = TelemetryPublisher(
        host=getattr(args, "telemetry_host", DEFAULT_TELEMETRY_HOST),
        port=getattr(args, "telemetry_port", DEFAULT_TELEMETRY_PORT),
        session_id=session_id, protocol="spout_training",
        enabled=not getattr(args, "no_telemetry", False),
    )
    telemetry_started = False
    telemetry_last_publish = None

    def telemetry_state(phase, training_index=None, next_reward_in_sec=None,
                        bait_index=None, bait_total=None, force=False):
        nonlocal telemetry_last_publish
        if not telemetry.enabled:
            return False
        now = time.monotonic()
        if not force and telemetry_last_publish is not None \
                and now - telemetry_last_publish < TELEMETRY_HEARTBEAT_SEC:
            return False
        completed_training_ids = completed_command_ids(
            all_events, attempted_training_reward_command_ids, "reward_complete")
        criterion = evaluate_training_criterion(
            successes, window=args.criterion_window,
            fraction=args.criterion_fraction,
        )
        state = {
            "phase": phase, "training_reward_index": training_index,
            "maximum_training_rewards": args.max_rewards,
            "criterion_window_rewards": args.criterion_window,
            "criterion_success_fraction": args.criterion_fraction,
            "reward_volume_ul": config["reward_volume_ul"],
            "reward_to_suction_delay_sec": DEFAULT_SUCTION_DELAY_SEC,
            "requested_bait_reward_count": requested_bait_reward_count,
            "planned_training_water_ul": len(interval_plan) * config["reward_volume_ul"],
            "telemetry_enabled": telemetry_started,
            "attempted_training_reward_count": attempted_training_reward_count,
            "completed_training_reward_count": len(completed_training_ids),
            "retrieval_success_count_session": sum(bool(value) for value in successes),
            "retrieval_failure_count_session": len(successes) - sum(bool(value) for value in successes),
            "recent_20_success_count": criterion["recent_success_count"],
            "recent_20_success_fraction": criterion["recent_success_fraction"],
            "criterion_evaluable": criterion["criterion_evaluable"],
            "training_passed": training_passed,
            "task_water_delivered_ul_session": len(completed_training_ids) * config["reward_volume_ul"],
            "bait_water_ul_session": len(completed_command_ids(
                all_events, attempted_bait_reward_command_ids, "reward_complete")) * config["reward_volume_ul"],
            "total_water_ul_session": (
                len(completed_training_ids) + len(completed_command_ids(
                    all_events, attempted_bait_reward_command_ids, "reward_complete"))
                ) * config["reward_volume_ul"],
            "total_lick_onset_count_session": sum(
                event.get("event_type") == "lick_onset" for event in all_events
            ),
            "next_reward_in_sec": next_reward_in_sec,
            "bait_index": bait_index, "bait_total": bait_total,
        }
        try:
            published = telemetry.publish_state(
                build_spout_state_payload(session_id, args.mouse_id, state)
            )
            telemetry_last_publish = now
            return published
        except Exception:
            return False

    try:
        event_path.touch()
        lick_path.touch()
        if not args.simulate_gpio:
            screen = prepare_gray_display(root)
        ready = client.start()
        metadata = {
            "protocol_name": "spout_training", "mouse_id": args.mouse_id,
            "session_id": session_id, "repository_commit": git_commit(),
            "repository_dirty": git_dirty(), "reward_pin_bcm": config["reward_pin_bcm"],
            "suction_pin_bcm": config["suction_pin_bcm"], "lick_pin_bcm": config["lick_pin_bcm"],
            "reward_num_pulses": config["reward_num_pulses"],
            "reward_pulse_on_sec": config["reward_pulse_on_sec"],
            "reward_pulse_off_sec": config["reward_pulse_off_sec"],
            "reward_volume_ul": config["reward_volume_ul"],
            "reward_to_suction_delay_sec": DEFAULT_SUCTION_DELAY_SEC,
            "suction_duration_sec": config["suction_duration_sec"],
            "requested_bait_reward_count": requested_bait_reward_count,
            "attempted_bait_reward_count": 0,
            "completed_bait_reward_count": 0,
            "completed_bait_suction_count": 0,
            "bait_reward_volume_ul": 0.0,
            "training_settle_sec": args.settle_sec,
            "reward_interval_min_sec": args.interval_min_sec,
            "reward_interval_max_sec": args.interval_max_sec, "schedule_seed": seed,
            "maximum_training_rewards": args.max_rewards,
            "minimum_rewards_for_criterion": DEFAULT_CRITERION_MIN_REWARDS,
            "criterion_window_rewards": args.criterion_window,
            "criterion_success_fraction": args.criterion_fraction,
            "planned_training_water_ul": args.max_rewards * config["reward_volume_ul"],
            "output_root": str(root), "gpio_worker_ready": bool(ready),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        telemetry_started = telemetry.start()
        try:
            telemetry.publish_session(build_spout_session_payload(
                session_id, args.mouse_id, {
                    **metadata, "phase": "STARTING",
                    "telemetry_enabled": telemetry_started,
                }
            ))
        except Exception:
            pass
        client.set_context(_context("spout_training_settle"))
        telemetry_state("STARTING", force=True)
        if not args.no_bait:
            for bait_index in range(1, requested_bait_reward_count + 1):
                bait_start = time.monotonic()
                attempted_bait_reward_count += 1
                telemetry_state("BAIT", bait_index, bait_index, requested_bait_reward_count, force=True)
                client.set_context(_context("spout_training_bait", bait_index))
                bait_context = _context("spout_training_bait", bait_index)
                bait_reward_id = client.trigger_reward(bait_context)
                attempted_bait_reward_command_ids.append(bait_reward_id)
                bait_reward_on_ns = None
                bait_reward_complete = False
                bait_reward_deadline = time.monotonic() + 5.0
                while time.monotonic() < bait_reward_deadline and (
                        bait_reward_on_ns is None or not bait_reward_complete):
                    for event in client.drain_events():
                        event.setdefault("phase", "spout_training_bait")
                        event.setdefault("training_reward_index", "")
                        event.setdefault("reward_command_id", bait_reward_id)
                        write_event(event)
                        if (event.get("command_id") == bait_reward_id
                                and event.get("event_type") == "reward_valve_on"):
                            bait_reward_on_ns = _event_ns(event)
                        if (event.get("command_id") == bait_reward_id
                                and event.get("event_type") == "reward_complete"):
                            bait_reward_complete = True
                    telemetry_state("BAIT", bait_index, bait_index, requested_bait_reward_count)
                    time.sleep(EPISODE_POLL_SEC)
                if bait_reward_on_ns is None:
                    raise RuntimeError(
                        "Bait reward %d did not produce reward_valve_on" % bait_index
                    )
                if bait_reward_complete:
                    completed_bait_reward_count += 1
                bait_suction_target_ns = bait_reward_on_ns + int(DEFAULT_SUCTION_DELAY_SEC * 1e9)
                while time.monotonic_ns() < bait_suction_target_ns:
                    for event in client.drain_events():
                        event.setdefault("phase", "spout_training_bait")
                        event.setdefault("training_reward_index", "")
                        event.setdefault("reward_command_id", bait_reward_id)
                        write_event(event)
                    telemetry_state("WAITING_SUCTION", bait_index, None, requested_bait_reward_count)
                    time.sleep(EPISODE_POLL_SEC)
                bait_suction_id = client.trigger_suction(bait_context)
                attempted_bait_suction_command_ids.append(bait_suction_id)
                bait_deadline = time.monotonic() + float(config["suction_duration_sec"]) + 0.5
                bait_suction_complete = False
                while time.monotonic() < bait_deadline:
                    for event in client.drain_events():
                        event.setdefault("phase", "spout_training_bait")
                        event.setdefault("training_reward_index", "")
                        event.setdefault("reward_command_id", bait_reward_id)
                        event.setdefault("suction_command_id", bait_suction_id)
                        write_event(event)
                        if (event.get("command_id") == bait_suction_id
                                and event.get("event_type") == "suction_complete"):
                            bait_suction_complete = True
                    telemetry_state("SUCTION", bait_index, None, requested_bait_reward_count)
                    time.sleep(EPISODE_POLL_SEC)
                if bait_suction_complete:
                    completed_bait_suction_count += 1
                else:
                    raise RuntimeError(
                        "Bait suction %d did not complete" % bait_index
                    )
                if not bait_reward_complete:
                    raise RuntimeError(
                        "Bait reward %d did not complete" % bait_index
                    )
                if bait_index < requested_bait_reward_count:
                    time.sleep(max(0.0, BAIT_INTERVAL_SEC - (time.monotonic() - bait_start)))
        client.set_context(_context("spout_training_settle"))
        telemetry_state("SETTLE", force=True)
        settle_deadline = time.monotonic() + max(0.0, float(args.settle_sec))
        while time.monotonic() < settle_deadline:
            for event in client.drain_events():
                event.setdefault("phase", "spout_training_settle")
                write_event(event)
            telemetry_state("SETTLE")
            time.sleep(EPISODE_POLL_SEC)
        schedule = anchor_training_schedule(time.monotonic_ns(), interval_plan)
        write_rows(planned_path, schedule, [
            "training_reward_index", "planned_interval_sec",
            "planned_reward_target_monotonic_ns",
        ])
        previous_actual_reward_on_ns = None
        for plan in schedule:
            if interrupted:
                break
            target = plan["planned_reward_target_monotonic_ns"]
            if previous_actual_reward_on_ns is not None:
                target = max(
                    target,
                    previous_actual_reward_on_ns
                    + int(round(plan["planned_interval_sec"] * 1e9)),
                )
            telemetry_state(
                "WAITING_REWARD", plan["training_reward_index"],
                max(0.0, (target - time.monotonic_ns()) / 1e9),
                force=True,
            )
            while time.monotonic_ns() < target:
                time.sleep(min(EPISODE_POLL_SEC, max(0.0, (target - time.monotonic_ns()) / 1e9)))
                for event in client.drain_events():
                    write_event(event)
                remaining_sec = max(0.0, (target - time.monotonic_ns()) / 1e9)
                # Keep the trigger boundary free of telemetry work.  The next
                # heartbeat will be emitted after the reward command instead.
                if remaining_sec > 0.05:
                    telemetry_state(
                        "WAITING_REWARD", plan["training_reward_index"],
                        remaining_sec,
                    )
            index = plan["training_reward_index"]
            context = _context("spout_training", index)
            attempted_training_reward_count += 1
            client.set_context(context)
            command_id = client.trigger_reward(context)
            attempted_training_reward_command_ids.append(command_id)
            command_ns = time.monotonic_ns()
            client.set_context(_context("spout_training_reward", index, command_id))
            telemetry_state("REWARD", index, force=True)
            reward_on_ns = None
            suction_on_ns = None
            suction_id = ""
            episode_deadline = time.monotonic() + DEFAULT_SUCTION_DELAY_SEC + float(config["suction_duration_sec"]) + 2.0
            episode_events = []
            while time.monotonic() < episode_deadline:
                events = client.drain_events()
                for event in events:
                    event.setdefault("training_reward_index", index)
                    event.setdefault("reward_command_id", command_id)
                    write_event(event)
                    episode_events.append(event)
                    if event.get("event_type") == "reward_valve_on" and reward_on_ns is None:
                        reward_on_ns = _event_ns(event)
                if reward_on_ns is not None and suction_id == "" and time.monotonic_ns() >= reward_on_ns + int(DEFAULT_SUCTION_DELAY_SEC * 1e9):
                    suction_target_ns = reward_on_ns + int(DEFAULT_SUCTION_DELAY_SEC * 1e9)
                    suction_id = client.trigger_suction(_context("spout_training", index, command_id))
                    attempted_training_suction_command_ids.append(suction_id)
                    suction_command_ns = time.monotonic_ns()
                    client.set_context(_context("spout_training_suction", index, command_id, suction_id))
                    telemetry_state("SUCTION", index, force=True)
                if suction_id and any(e.get("event_type") == "suction_on" and e.get("command_id") == suction_id for e in episode_events):
                    suction_on_ns = next(e.get("monotonic_ns") for e in episode_events if e.get("event_type") == "suction_on" and e.get("command_id") == suction_id)
                if suction_id and any(e.get("event_type") == "suction_complete" and e.get("command_id") == suction_id for e in episode_events):
                    break
                telemetry_state("SUCTION" if suction_id else "REWARD", index)
                time.sleep(EPISODE_POLL_SEC)
            if reward_on_ns is None or suction_on_ns is None:
                raise RuntimeError("Incomplete reward/suction episode %d" % index)
            reward_complete = first_command_event(
                all_events, "reward_complete", command_id)
            suction_complete = first_command_event(
                all_events, "suction_complete", suction_id)
            if reward_complete is None or suction_complete is None:
                raise RuntimeError("Incomplete reward/suction episode %d" % index)
            metrics = compute_reward_lick_metrics(all_events, reward_on_ns, suction_on_ns)
            previous_actual_reward_on_ns = reward_on_ns
            successes.append(metrics["retrieval_success"])
            criterion = evaluate_training_criterion(successes, window=args.criterion_window, fraction=args.criterion_fraction)
            training_passed = criterion["criterion_passed"]
            if training_passed and pass_index is None:
                pass_index = index
            reward_received = first_command_event(
                all_events, "reward_command_received", command_id)
            reward_on = first_command_event(
                all_events, "reward_valve_on", command_id)
            reward_off = first_command_event(
                all_events, "reward_valve_off", command_id)
            suction_received = first_command_event(
                all_events, "suction_command_received", suction_id)
            suction_on = first_command_event(
                all_events, "suction_on", suction_id)
            suction_off = first_command_event(
                all_events, "suction_off", suction_id)
            reward_rows.append(dict(
                plan, **metrics,
                reward_command_id=command_id,
                reward_command_unix_ns=_event_timestamp_fields(reward_received).get("unix_ns", ""),
                reward_command_monotonic_ns=_event_timestamp_fields(reward_received).get("monotonic_ns", command_ns),
                reward_on_unix_ns=_event_timestamp_fields(reward_on).get("unix_ns", ""),
                reward_on_monotonic_ns=_event_timestamp_fields(reward_on).get("monotonic_ns", reward_on_ns),
                reward_off_unix_ns=_event_timestamp_fields(reward_off).get("unix_ns", ""),
                reward_off_monotonic_ns=_event_timestamp_fields(reward_off).get("monotonic_ns", ""),
                reward_complete_unix_ns=_event_timestamp_fields(reward_complete).get("unix_ns", ""),
                reward_complete_monotonic_ns=_event_timestamp_fields(reward_complete).get("monotonic_ns", ""),
                software_reward_timing_error_sec=(
                    (reward_on_ns - target) / 1e9
                ),
                effective_reward_target_monotonic_ns=target,
                suction_command_id=suction_id,
                suction_target_monotonic_ns=suction_target_ns,
                suction_on_unix_ns=_event_timestamp_fields(suction_on).get("unix_ns", ""),
                suction_on_monotonic_ns=_event_timestamp_fields(suction_on).get("monotonic_ns", suction_on_ns),
                suction_off_unix_ns=_event_timestamp_fields(suction_off).get("unix_ns", ""),
                suction_off_monotonic_ns=_event_timestamp_fields(suction_off).get("monotonic_ns", ""),
                suction_complete_unix_ns=_event_timestamp_fields(suction_complete).get("unix_ns", ""),
                suction_complete_monotonic_ns=_event_timestamp_fields(suction_complete).get("monotonic_ns", ""),
                software_suction_delay_sec=(suction_on_ns - reward_on_ns) / 1e9,
                recent_20_success_count=criterion["recent_success_count"],
                recent_20_success_fraction=criterion["recent_success_fraction"],
                criterion_evaluable=criterion["criterion_evaluable"],
                criterion_passed_after_this_reward=training_passed,
            ))
            print("Reward %d/%d | recent20: %s" % (
                index, args.max_rewards,
                "%.0f%%" % (100 * criterion["recent_success_fraction"]) if criterion["criterion_evaluable"] else "not yet evaluable",
            ))
            try:
                telemetry.publish_trial(
                    build_spout_trial_payload(
                        session_id, args.mouse_id, reward_rows[-1],
                        {
                            "phase": "REWARD", "maximum_training_rewards": args.max_rewards,
                            "criterion_window_rewards": args.criterion_window,
                            "criterion_success_fraction": args.criterion_fraction,
                            "reward_volume_ul": config["reward_volume_ul"],
                            "reward_to_suction_delay_sec": DEFAULT_SUCTION_DELAY_SEC,
                            "attempted_training_reward_count": attempted_training_reward_count,
                            "completed_training_reward_count": len(completed_command_ids(
                                all_events, attempted_training_reward_command_ids, "reward_complete")),
                            "retrieval_success_count_session": sum(bool(value) for value in successes),
                            "retrieval_failure_count_session": len(successes) - sum(bool(value) for value in successes),
                            "task_water_delivered_ul_session": len(completed_command_ids(
                                all_events, attempted_training_reward_command_ids, "reward_complete")) * config["reward_volume_ul"],
                            "bait_water_ul_session": len(completed_command_ids(
                                all_events, attempted_bait_reward_command_ids, "reward_complete")) * config["reward_volume_ul"],
                            "total_water_ul_session": (len(completed_command_ids(
                                all_events, attempted_training_reward_command_ids, "reward_complete"))
                                + len(completed_command_ids(all_events, attempted_bait_reward_command_ids, "reward_complete"))) * config["reward_volume_ul"],
                            "total_lick_onset_count_session": sum(
                                event.get("event_type") == "lick_onset" for event in all_events),
                            "training_passed": training_passed,
                        },
                    )
                )
            except Exception:
                pass
            if training_passed:
                print("Training criterion reached.")
                break
            client.set_context(_context("spout_training_inter_reward", index))
    except KeyboardInterrupt:
        interrupted = True
        failure_summary = "operator interrupted"
    except Exception as exc:
        failure_exc = exc
        failure_summary = "%s: %s" % (type(exc).__name__, str(exc))
    finally:
        signal.signal(signal.SIGINT, old_handler)
        try:
            for event in client.drain_events():
                write_event(event)
        finally:
            for event in client.shutdown():
                write_event(event)
            if screen is not None:
                close = getattr(screen, "close", None)
                if close is not None:
                    close()
    criterion = evaluate_training_criterion(successes, window=args.criterion_window, fraction=args.criterion_fraction)
    write_rows(summary_path, reward_rows, REWARD_SUMMARY_FIELDS)
    completed_bait_reward_count = len(completed_command_ids(
        all_events, attempted_bait_reward_command_ids, "reward_complete"))
    completed_bait_suction_count = len(completed_command_ids(
        all_events, attempted_bait_suction_command_ids, "suction_complete"))
    completed_training_reward_command_ids = completed_command_ids(
        all_events, attempted_training_reward_command_ids, "reward_complete")
    completed_training_suction_command_ids = completed_command_ids(
        all_events, attempted_training_suction_command_ids, "suction_complete")
    qc = build_training_qc(
        reward_rows, all_events, training_passed,
        dict(criterion, window=args.criterion_window),
        attempted_reward_command_ids=attempted_training_reward_command_ids,
        attempted_suction_command_ids=attempted_training_suction_command_ids,
    )
    qc.update({"operator_interrupted": interrupted,
               "training_passed": training_passed,
               "requested_bait_reward_count": requested_bait_reward_count,
               "attempted_bait_reward_count": attempted_bait_reward_count,
               "completed_bait_reward_count": completed_bait_reward_count,
               "attempted_bait_suction_count": len(attempted_bait_suction_command_ids),
               "completed_bait_suction_count": completed_bait_suction_count,
               "bait_hardware_complete": (
                   attempted_bait_reward_count == completed_bait_reward_count
                   and len(attempted_bait_suction_command_ids) == completed_bait_suction_count
               )})
    finalize_training_qc(
        qc, not interrupted and failure_exc is None,
        attempted_bait_reward_count, completed_bait_reward_count,
        len(attempted_bait_suction_command_ids), completed_bait_suction_count,
    )
    qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True))
    metadata.update({
        "requested_bait_reward_count": requested_bait_reward_count,
        "attempted_bait_reward_count": attempted_bait_reward_count,
        "completed_bait_reward_count": completed_bait_reward_count,
        "attempted_bait_suction_count": len(attempted_bait_suction_command_ids),
        "completed_bait_suction_count": completed_bait_suction_count,
        "bait_hardware_complete": (
            attempted_bait_reward_count == completed_bait_reward_count
            and len(attempted_bait_suction_command_ids) == completed_bait_suction_count
        ),
        "planned_schedule_length": len(schedule) or len(interval_plan),
        "attempted_training_reward_count": attempted_training_reward_count,
        "completed_training_reward_count": len(completed_training_reward_command_ids),
        "completed_suction_count": len(completed_training_suction_command_ids),
        "total_lick_onset_count": sum(1 for event in all_events if event.get("event_type") == "lick_onset"),
        "training_passed": training_passed, "training_pass_reward_index": pass_index,
        "final_recent_20_success_count": criterion["recent_success_count"],
        "final_recent_20_success_fraction": criterion["recent_success_fraction"],
        "planned_training_water_ul": len(interval_plan) * config["reward_volume_ul"],
        "actual_training_water_ul": len(completed_training_reward_command_ids) * config["reward_volume_ul"],
        "bait_water_ul": completed_bait_reward_count * config["reward_volume_ul"],
        "total_water_ul": (len(completed_training_reward_command_ids) + completed_bait_reward_count) * config["reward_volume_ul"],
        "session_completed": not interrupted and failure_exc is None,
        "operator_interrupted": interrupted,
        "failure_summary": failure_summary,
        "session_qc_json": str(qc_path), "reward_summary_csv": str(summary_path),
        "planned_rewards_csv": str(planned_path), "event_log_csv": str(event_path),
        "lick_events_csv": str(lick_path),
        "qc_pass": qc["qc_pass"],
        "qc_fail_reasons": qc["qc_fail_reasons"],
        "telemetry_enabled": telemetry_started,
        "telemetry_host": getattr(args, "telemetry_host", DEFAULT_TELEMETRY_HOST),
        "telemetry_port": getattr(args, "telemetry_port", DEFAULT_TELEMETRY_PORT),
    })
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    final_phase = "COMPLETE"
    if interrupted:
        final_phase = "INTERRUPTED"
    elif failure_exc is not None:
        final_phase = "FAILED"
    final_payload = build_spout_state_payload(
        session_id, args.mouse_id, {
            **metadata, "phase": final_phase,
            "training_reward_index": pass_index,
            "maximum_training_rewards": args.max_rewards,
            "criterion_window_rewards": args.criterion_window,
            "criterion_success_fraction": args.criterion_fraction,
            "reward_to_suction_delay_sec": DEFAULT_SUCTION_DELAY_SEC,
            "failure_summary": failure_summary,
        },
    )
    try:
        telemetry.publish_state(final_payload)
    except Exception:
        pass
    try:
        telemetry.close()
    except Exception:
        pass
    if not training_passed:
        print("Training criterion not reached.\nRepeat spout training before image conditioning.")
    if failure_exc is not None:
        raise failure_exc
    return metadata, qc


def main(argv=None):
    args = parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
