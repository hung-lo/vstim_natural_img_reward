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
    "software_reward_timing_error_sec", "suction_command_id",
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


def build_training_schedule(start_monotonic_ns, max_rewards=DEFAULT_MAX_REWARDS,
                            interval_min_sec=DEFAULT_INTERVAL_MIN_SEC,
                            interval_max_sec=DEFAULT_INTERVAL_MAX_SEC,
                            seed=None):
    """Generate all reward targets before any scheduled reward is delivered."""
    max_rewards = int(max_rewards)
    low, high = float(interval_min_sec), float(interval_max_sec)
    if max_rewards < 1:
        raise ValueError("max_rewards must be at least 1")
    if low <= 0 or high < low:
        raise ValueError("Require 0 < interval_min_sec <= interval_max_sec")
    rng = random.Random(seed)
    target_ns = int(start_monotonic_ns)
    rows = []
    previous_target_ns = None
    for index in range(1, max_rewards + 1):
        interval = 0.0 if previous_target_ns is None else rng.uniform(low, high)
        if previous_target_ns is not None:
            target_ns += int(round(interval * 1_000_000_000.0))
        rows.append({
            "training_reward_index": index,
            "planned_interval_sec": interval,
            "planned_reward_target_monotonic_ns": target_ns,
        })
        previous_target_ns = target_ns
    return rows


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


def build_training_qc(reward_rows, events, training_passed, criterion):
    """Return system QC separately from the mouse's behavioral outcome."""
    reward_ids = {row.get("reward_command_id") for row in reward_rows if row.get("reward_command_id")}
    suction_ids = {row.get("suction_command_id") for row in reward_rows if row.get("suction_command_id")}
    def count(types, ids):
        return sum(1 for event in events if event.get("command_id") in ids and event.get("event_type") in types)
    expected_pulses = len(reward_rows)
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
        "planned_scheduled_training_reward_count": len(reward_rows),
        "reward_command_received_count": count({"reward_command_received"}, reward_ids),
        "reward_complete_count": count({"reward_complete"}, reward_ids),
        "reward_valve_on_count": count({"reward_valve_on"}, reward_ids),
        "reward_valve_off_count": count({"reward_valve_off"}, reward_ids),
        "planned_suction_count": len(reward_rows),
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
        and checks["suction_command_received_count"] == expected_pulses
        and checks["suction_on_count"] == expected_pulses
        and checks["suction_off_count"] == expected_pulses
        and checks["suction_complete_count"] == expected_pulses
        and not checks["missing_reward_command_ids"]
        and not checks["unexpected_reward_command_ids"]
        and not checks["duplicate_reward_command_ids"]
        and not checks["incomplete_reward_command_ids"]
        and not checks["missing_suction_command_ids"]
        and not checks["unexpected_suction_command_ids"]
        and not checks["duplicate_suction_command_ids"]
        and not checks["incomplete_suction_command_ids"]
    )
    return checks


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
    session_start_ns = time.monotonic_ns()
    schedule = build_training_schedule(
        session_start_ns + int(float(args.settle_sec) * 1e9), args.max_rewards,
        args.interval_min_sec, args.interval_max_sec, seed,
    )
    write_rows(planned_path, schedule, [
        "training_reward_index", "planned_interval_sec",
        "planned_reward_target_monotonic_ns",
    ])
    all_events, reward_rows, successes = [], [], []
    bait_count = 0
    interrupted = False
    training_passed = False
    pass_index = None
    old_handler = signal.getsignal(signal.SIGINT)

    def write_event(event):
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
            "bait_reward_count": 0, "bait_reward_volume_ul": 0.0,
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
        client.set_context(_context("spout_training_settle"))
        if not args.no_bait:
            bait_count = args.bait_drops
            for bait_index in range(1, bait_count + 1):
                bait_start = time.monotonic()
                client.set_context(_context("spout_training_bait", bait_index))
                bait_context = _context("spout_training_bait", bait_index)
                bait_reward_id = client.trigger_reward(bait_context)
                while time.monotonic() < bait_start + DEFAULT_SUCTION_DELAY_SEC:
                    for event in client.drain_events():
                        event.setdefault("phase", "spout_training_bait")
                        event.setdefault("training_reward_index", "")
                        event.setdefault("reward_command_id", bait_reward_id)
                        write_event(event)
                    time.sleep(EPISODE_POLL_SEC)
                bait_suction_id = client.trigger_suction(bait_context)
                bait_deadline = time.monotonic() + float(config["suction_duration_sec"]) + 0.5
                while time.monotonic() < bait_deadline:
                    for event in client.drain_events():
                        event.setdefault("phase", "spout_training_bait")
                        event.setdefault("training_reward_index", "")
                        event.setdefault("reward_command_id", bait_reward_id)
                        event.setdefault("suction_command_id", bait_suction_id)
                        write_event(event)
                    time.sleep(EPISODE_POLL_SEC)
                if bait_index < bait_count:
                    time.sleep(max(0.0, BAIT_INTERVAL_SEC - (time.monotonic() - bait_start)))
        time.sleep(max(0.0, float(args.settle_sec)))
        for plan in schedule:
            if interrupted:
                break
            target = plan["planned_reward_target_monotonic_ns"]
            while time.monotonic_ns() < target:
                time.sleep(min(EPISODE_POLL_SEC, max(0.0, (target - time.monotonic_ns()) / 1e9)))
                for event in client.drain_events():
                    write_event(event)
            index = plan["training_reward_index"]
            context = _context("spout_training", index)
            client.set_context(context)
            command_id = client.trigger_reward(context)
            command_ns = time.monotonic_ns()
            client.set_context(_context("spout_training_reward", index, command_id))
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
                    suction_command_ns = time.monotonic_ns()
                    client.set_context(_context("spout_training_suction", index, command_id, suction_id))
                if suction_id and any(e.get("event_type") == "suction_on" and e.get("command_id") == suction_id for e in episode_events):
                    suction_on_ns = next(e.get("monotonic_ns") for e in episode_events if e.get("event_type") == "suction_on" and e.get("command_id") == suction_id)
                if suction_id and any(e.get("event_type") == "suction_complete" and e.get("command_id") == suction_id for e in episode_events):
                    break
                time.sleep(EPISODE_POLL_SEC)
            if reward_on_ns is None or suction_on_ns is None:
                raise RuntimeError("Incomplete reward/suction episode %d" % index)
            metrics = compute_reward_lick_metrics(episode_events, reward_on_ns, suction_on_ns)
            successes.append(metrics["retrieval_success"])
            criterion = evaluate_training_criterion(successes, window=args.criterion_window, fraction=args.criterion_fraction)
            training_passed = criterion["criterion_passed"]
            if training_passed and pass_index is None:
                pass_index = index
            reward_rows.append(dict(plan, **metrics, reward_command_id=command_id,
                                    reward_command_monotonic_ns=command_ns,
                                    suction_command_id=suction_id,
                                    suction_target_monotonic_ns=suction_target_ns,
                                    recent_20_success_count=criterion["recent_success_count"],
                                    recent_20_success_fraction=criterion["recent_success_fraction"],
                                    criterion_evaluable=criterion["criterion_evaluable"],
                                    criterion_passed_after_this_reward=training_passed))
            print("Reward %d/%d | recent20: %s" % (
                index, args.max_rewards,
                "%.0f%%" % (100 * criterion["recent_success_fraction"]) if criterion["criterion_evaluable"] else "not yet evaluable",
            ))
            if training_passed:
                print("Training criterion reached.")
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        signal.signal(signal.SIGINT, old_handler)
        try:
            for event in client.drain_events():
                write_event(event)
        finally:
            client.shutdown()
            if screen is not None:
                close = getattr(screen, "close", None)
                if close is not None:
                    close()
    criterion = evaluate_training_criterion(successes, window=args.criterion_window, fraction=args.criterion_fraction)
    write_rows(summary_path, reward_rows, REWARD_SUMMARY_FIELDS)
    qc = build_training_qc(reward_rows, all_events, training_passed, dict(criterion, window=args.criterion_window))
    qc.update({"operator_interrupted": interrupted, "session_completed": not interrupted,
               "training_passed": training_passed})
    qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True))
    metadata.update({
        "bait_reward_count": bait_count, "scheduled_training_reward_count": len(schedule),
        "completed_training_reward_count": len(reward_rows),
        "completed_suction_count": sum(1 for row in reward_rows if row.get("suction_command_id")),
        "total_lick_onset_count": sum(1 for event in all_events if event.get("event_type") == "lick_onset"),
        "training_passed": training_passed, "training_pass_reward_index": pass_index,
        "final_recent_20_success_count": criterion["recent_success_count"],
        "final_recent_20_success_fraction": criterion["recent_success_fraction"],
        "planned_training_water_ul": len(schedule) * config["reward_volume_ul"],
        "actual_training_water_ul": len(reward_rows) * config["reward_volume_ul"],
        "bait_water_ul": bait_count * config["reward_volume_ul"],
        "total_water_ul": (len(reward_rows) + bait_count) * config["reward_volume_ul"],
        "session_completed": not interrupted, "operator_interrupted": interrupted,
        "failure_summary": "" if not interrupted else "operator interrupted",
        "session_qc_json": str(qc_path), "reward_summary_csv": str(summary_path),
        "planned_rewards_csv": str(planned_path), "event_log_csv": str(event_path),
        "lick_events_csv": str(lick_path),
        "qc_pass": qc["qc_pass"],
    })
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    if not training_passed:
        print("Training criterion not reached.\nRepeat spout training before image conditioning.")
    return metadata, qc


def main(argv=None):
    args = parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
