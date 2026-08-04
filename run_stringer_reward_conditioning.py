#!/usr/bin/env python3
"""Natural-image Pavlovian reward-conditioning task for the behavior RPi.

This is a new entry point for ``vstim_natural``.  It deliberately does not
modify the existing working natural-image runners.

Protocol
--------
* 14 fixed natural images per mouse.
* 10 low-probability images at 2% each.
* 2 high-probability unrewarded images at 20% each.
* 2 high-probability rewarded images at 20% each.
* 1.5 s image duration.
* Open-loop reward at the 1.0 s image boundary on exactly 90% of
  rewarded-cue trials, independent of licking.
* Variable gray ITI.
* Matched gray PRE and POST background epochs with the photodiode patch off.

The 1.5 s image is rendered as two visually identical RPG raw sequences
(1.0 s and 0.5 s).  The parent sends the open-loop reward command between the
segments.  Reward pulse timing and lick capture run in a separate process; see
``reward_conditioning_gpio.py`` for the RPG/GIL rationale.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import run_stringer_vstim as base

from reward_conditioning_gpio import BehaviorGPIOClient
from reward_conditioning_protocol import (
    ALL_ROLES,
    REWARDED_HIGH_ROLES,
    REWARD_PROBABILITY,
    TRIALS_PER_BLOCK,
    create_or_load_assignment,
    global_panel_path,
    make_trial_plan,
    summarize_trial_plan,
)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path("/mnt/hd")
ASSIGNMENT_DIR = OUTPUT_ROOT / "vstim_reward_assignments"
SESSION_SUFFIX = "vstim_natural_reward_conditioning"
DEFAULT_HARDWARE_CONFIG_PATH = PROJECT_ROOT / "reward_conditioning_config.json"

STIM_DURATION_SEC = 1.5
REWARD_DELAY_SEC = 1.0
POST_REWARD_STIM_SEC = STIM_DURATION_SEC - REWARD_DELAY_SEC
DEFAULT_N_BLOCKS = 10
DEFAULT_ITI_MIN_SEC = 2.5
DEFAULT_ITI_MAX_SEC = 4.5
DEFAULT_PRE_BACKGROUND_MIN = 5.0
DEFAULT_POST_BACKGROUND_MIN = 5.0
BACKGROUND_POLL_SEC = 0.10

PLANNED_SEQUENCE_FIELDS = [
    "trial_index",
    "trial_number",
    "block_index",
    "block_number",
    "within_block_index",
    "within_block_number",
    "image_role",
    "image_category",
    "presentation_probability",
    "image_id",
    "image_filename",
    "image_path",
    "reward_eligible",
    "reward_scheduled",
    "reward_omission_scheduled",
    "rewarded_cue_presentation_ordinal",
    "planned_stim_duration_sec",
    "planned_reward_delay_sec",
    "planned_post_reward_stim_sec",
    "planned_iti_duration_sec",
]

EVENT_FIELDS = [
    "utc_iso",
    "unix_time_utc_sec",
    "unix_time_ns",
    "monotonic_ns",
    "event_type",
    "phase",
    "trial_index",
    "trial_number",
    "block_index",
    "block_number",
    "within_block_index",
    "within_block_number",
    "image_role",
    "image_category",
    "presentation_probability",
    "image_id",
    "image_filename",
    "reward_eligible",
    "reward_scheduled",
    "reward_omission_scheduled",
    "rewarded_cue_presentation_ordinal",
    "segment_name",
    "raw_path",
    "planned_duration_sec",
    "planned_iti_duration_sec",
    "display_request_unix_ns",
    "display_return_unix_ns",
    "display_request_perf_counter_ns",
    "display_return_perf_counter_ns",
    "display_call_duration_sec",
    "start_time_unix",
    "mean_interframe_us",
    "stddev_interframe_us",
    "segment_boundary_gap_sec",
    "command_id",
    "reward_pin_bcm",
    "lick_pin_bcm",
    "lick_edge",
    "reward_pulse_on_sec",
    "reward_pulse_off_sec",
    "reward_pulse_index",
    "reward_num_pulses",
    "reward_trigger_source",
    "notes",
]

TRIAL_SUMMARY_FIELDS = [
    "trial_index",
    "trial_number",
    "block_number",
    "image_role",
    "image_category",
    "image_id",
    "image_filename",
    "reward_eligible",
    "reward_scheduled",
    "reward_omission_scheduled",
    "planned_iti_duration_sec",
    "stim_request_unix_ns",
    "stim_segment1_return_unix_ns",
    "stim_segment2_request_unix_ns",
    "stim_offset_request_unix_ns",
    "segment_boundary_gap_sec",
    "first_reward_valve_on_unix_ns",
    "actual_reward_delay_from_software_stim_request_sec",
    "lick_count_pre_0p5_sec",
    "lick_count_0_to_0p5_sec",
    "lick_count_0_to_1p0_sec",
    "lick_count_0p5_to_1p0_sec",
    "lick_count_1p0_to_1p5_sec",
    "lick_count_1p5_to_2p0_sec",
    "notes",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware-config",
        default=str(DEFAULT_HARDWARE_CONFIG_PATH),
        help="JSON file containing reward/lick GPIO and calibrated pulse timing.",
    )
    parser.add_argument(
        "--simulate-gpio",
        action="store_true",
        help="Run the GPIO child with a mock valve and no lick input.",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Do not offer remote face-camera recording.",
    )
    return parser.parse_args(argv)


def prompt_float_or_default(prompt, default_value, minimum=None, maximum=None):
    raw = base.prompt_text("%s [%s]: " % (prompt, default_value)).strip()
    value = float(default_value) if not raw else float(raw)
    if minimum is not None and value < minimum:
        raise ValueError("%s must be at least %s" % (prompt, minimum))
    if maximum is not None and value > maximum:
        raise ValueError("%s must be at most %s" % (prompt, maximum))
    return value


def make_session_name(mouse_id, session_stamp):
    return "%s_%s_%s" % (mouse_id, session_stamp, SESSION_SUFFIX)


def exact_timestamp_event(event_type, **fields):
    captured = base.capture_timestamp()
    row = {
        "utc_iso": captured["utc_iso"],
        "unix_time_utc_sec": captured["unix_sec"],
        "unix_time_ns": captured["unix_ns"],
        "monotonic_ns": time.monotonic_ns(),
        "event_type": event_type,
    }
    row.update(fields)
    return row


def append_event(event_log_path, row):
    base.append_csv_row(event_log_path, row, EVENT_FIELDS)


def trial_context(trial, phase):
    return {
        "phase": phase,
        "trial_index": trial.get("trial_index", ""),
        "trial_number": trial.get("trial_number", ""),
        "block_index": trial.get("block_index", ""),
        "block_number": trial.get("block_number", ""),
        "within_block_index": trial.get("within_block_index", ""),
        "within_block_number": trial.get("within_block_number", ""),
        "image_role": trial.get("image_role", ""),
        "image_category": trial.get("image_category", ""),
        "presentation_probability": trial.get("presentation_probability", ""),
        "image_id": trial.get("image_id", ""),
        "image_filename": trial.get("image_filename", ""),
        "reward_eligible": trial.get("reward_eligible", ""),
        "reward_scheduled": trial.get("reward_scheduled", ""),
        "reward_omission_scheduled": trial.get("reward_omission_scheduled", ""),
        "rewarded_cue_presentation_ordinal": trial.get(
            "rewarded_cue_presentation_ordinal", ""
        ),
    }


def background_context(phase):
    return {
        "phase": phase,
        "trial_index": "",
        "trial_number": "",
        "block_index": "",
        "block_number": "",
        "within_block_index": "",
        "within_block_number": "",
        "image_role": "",
        "image_category": "",
        "presentation_probability": "",
        "image_id": "",
        "image_filename": "",
        "reward_eligible": "",
        "reward_scheduled": "",
        "reward_omission_scheduled": "",
        "rewarded_cue_presentation_ordinal": "",
    }


def load_hardware_config(path, simulate_gpio=False):
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            "Missing hardware configuration: %s. Copy "
            "reward_conditioning_config.example.json to "
            "reward_conditioning_config.json and enter the calibrated "
            "reward_pulse_on_sec from the working Go/NoGo setup." % path
        )
    config = json.loads(path.read_text())
    required = [
        "reward_pin_bcm",
        "lick_pin_bcm",
        "reward_pulse_on_sec",
        "reward_pulse_off_sec",
        "reward_num_pulses",
    ]
    missing = [name for name in required if name not in config]
    if missing:
        raise RuntimeError("Hardware config is missing: %s" % ", ".join(missing))

    config["simulate_gpio"] = bool(simulate_gpio or config.get("simulate_gpio", False))
    if config["simulate_gpio"] and config.get("reward_pulse_on_sec") in (None, ""):
        config["reward_pulse_on_sec"] = 0.002
    if config.get("reward_pulse_on_sec") in (None, ""):
        raise RuntimeError(
            "reward_pulse_on_sec is unset. Copy the calibrated "
            "session_info['solenoid_blink_duration'] value from the working "
            "Go/NoGo setup; do not guess this valve timing."
        )

    config["reward_pin_bcm"] = int(config["reward_pin_bcm"])
    config["lick_pin_bcm"] = int(config["lick_pin_bcm"])
    config["reward_pulse_on_sec"] = float(config["reward_pulse_on_sec"])
    config["reward_pulse_off_sec"] = float(config["reward_pulse_off_sec"])
    config["reward_num_pulses"] = int(config["reward_num_pulses"])
    if config["reward_pin_bcm"] == config["lick_pin_bcm"]:
        raise RuntimeError("Reward and lick GPIO pins cannot be identical.")
    if config["reward_pulse_on_sec"] <= 0:
        raise RuntimeError("reward_pulse_on_sec must be positive.")
    if config["reward_pulse_off_sec"] < 0:
        raise RuntimeError("reward_pulse_off_sec cannot be negative.")
    if config["reward_num_pulses"] < 1:
        raise RuntimeError("reward_num_pulses must be at least 1.")
    return config


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def build_background_raw(rpg_module, raw_cache_root):
    background_dir = base.ensure_dir(raw_cache_root / "background")
    canvas = base.build_canvas(None, base.SCREEN_RESOLUTION, photodiode_on=False)
    path = background_dir / "gray_photodiode_off_one_frame.raw"
    base.convert_canvas_to_rpg_raw(
        rpg_module,
        canvas,
        path,
        1.0 / float(base.REFRESH_RATE_HZ),
    )
    return path


def build_stimulus_segment_raws(rpg_module, raw_cache_root, assignment_rows):
    first_dir = base.ensure_dir(raw_cache_root / "stim_first_1p0_sec")
    second_dir = base.ensure_dir(raw_cache_root / "stim_second_0p5_sec")
    raw_paths = {}
    for row in assignment_rows:
        image_path = Path(row["image_path"])
        canvas = base.build_canvas(
            image_path, base.SCREEN_RESOLUTION, photodiode_on=True
        )
        first_path = first_dir / (image_path.stem + "_first_1p0s.raw")
        second_path = second_dir / (image_path.stem + "_second_0p5s.raw")
        base.convert_canvas_to_rpg_raw(
            rpg_module, canvas, first_path, REWARD_DELAY_SEC
        )
        base.convert_canvas_to_rpg_raw(
            rpg_module, canvas, second_path, POST_REWARD_STIM_SEC
        )
        raw_paths[image_path.name] = {
            "first": first_path,
            "second": second_path,
        }
    return raw_paths


def make_display_event(event_type, trial, segment_name, raw_path, planned_duration_sec, perf, timing):
    row = trial_context(trial, "stimulus")
    row.update(
        {
            "utc_iso": timing["request_utc_iso"],
            "unix_time_utc_sec": timing["request_unix_sec"],
            "unix_time_ns": timing["request_unix_ns"],
            "monotonic_ns": timing["request_perf_counter_ns"],
            "event_type": event_type,
            "segment_name": segment_name,
            "raw_path": str(raw_path),
            "planned_duration_sec": planned_duration_sec,
            "planned_iti_duration_sec": trial["planned_iti_duration_sec"],
            "display_request_unix_ns": timing["request_unix_ns"],
            "display_return_unix_ns": timing["return_unix_ns"],
            "display_request_perf_counter_ns": timing["request_perf_counter_ns"],
            "display_return_perf_counter_ns": timing["return_perf_counter_ns"],
            "display_call_duration_sec": "%.9f" % timing["duration_sec"],
            "start_time_unix": getattr(perf, "start_time", ""),
            "mean_interframe_us": getattr(perf, "mean_interframe", ""),
            "stddev_interframe_us": getattr(perf, "stddev_interframe", ""),
        }
    )
    return row


def log_drained_gpio_events(gpio_client, event_log_path, all_gpio_events):
    events = gpio_client.drain_events()
    for event in events:
        append_event(event_log_path, event)
        if event.get("message_type") == "event":
            all_gpio_events.append(dict(event))
    return events


def wait_until(deadline_monotonic, gpio_client, event_log_path, all_gpio_events):
    while True:
        remaining = deadline_monotonic - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(BACKGROUND_POLL_SEC, remaining))
        log_drained_gpio_events(gpio_client, event_log_path, all_gpio_events)


def hold_background(
    phase,
    duration_sec,
    gpio_client,
    event_log_path,
    all_gpio_events,
):
    gpio_client.set_context(background_context(phase))
    start = exact_timestamp_event(
        phase + "_start",
        phase=phase,
        planned_duration_sec=float(duration_sec),
        notes="gray_127; photodiode_patch_off",
    )
    append_event(event_log_path, start)
    start_monotonic = time.monotonic()
    wait_until(
        start_monotonic + float(duration_sec),
        gpio_client,
        event_log_path,
        all_gpio_events,
    )
    actual = time.monotonic() - start_monotonic
    end = exact_timestamp_event(
        phase + "_end",
        phase=phase,
        planned_duration_sec=float(duration_sec),
        notes="actual_duration_sec=%.9f" % actual,
    )
    append_event(event_log_path, end)
    return actual


def run_trials(
    screen,
    trials,
    loaded_raws,
    loaded_background_raw,
    raw_paths,
    background_raw_path,
    gpio_client,
    event_log_path,
    all_gpio_events,
):
    runtime_by_trial = {}
    task_start_monotonic = time.monotonic()
    total_trials = len(trials)

    for completed_count, trial in enumerate(trials, start=1):
        context = trial_context(trial, "stimulus")
        gpio_client.set_context(context)
        log_drained_gpio_events(gpio_client, event_log_path, all_gpio_events)

        image_raws = loaded_raws[trial["image_filename"]]
        image_paths = raw_paths[trial["image_filename"]]

        first_perf, first_timing = base.display_raw_with_timing(
            screen, image_raws["first"]
        )

        # Do not write to disk at the 1.0 s boundary.  Send the precomputed
        # open-loop command, if any, then immediately start the identical 0.5 s
        # segment.  Lick state is never inspected here.
        boundary_timestamp = base.capture_timestamp()
        boundary_monotonic_ns = time.monotonic_ns()
        reward_command_id = ""
        if trial["reward_scheduled"]:
            reward_command_id = gpio_client.trigger_reward(context)

        second_perf, second_timing = base.display_raw_with_timing(
            screen, image_raws["second"]
        )
        segment_boundary_gap_sec = (
            second_timing["request_perf_counter_ns"]
            - first_timing["return_perf_counter_ns"]
        ) / 1_000_000_000.0

        # Turn the image/photodiode patch off before CSV writes or context ACKs.
        iti_start_monotonic = time.monotonic()
        background_perf, background_timing = base.display_raw_with_timing(
            screen, loaded_background_raw
        )
        iti_deadline = iti_start_monotonic + float(trial["planned_iti_duration_sec"])

        gpio_client.set_context(trial_context(trial, "iti"))

        first_row = make_display_event(
            "stimulus_segment_1",
            trial,
            "first_1p0_sec",
            image_paths["first"],
            REWARD_DELAY_SEC,
            first_perf,
            first_timing,
        )
        first_row["segment_boundary_gap_sec"] = "%.9f" % segment_boundary_gap_sec
        append_event(event_log_path, first_row)

        boundary_row = trial_context(trial, "stimulus")
        boundary_row.update(
            {
                "utc_iso": boundary_timestamp["utc_iso"],
                "unix_time_utc_sec": boundary_timestamp["unix_sec"],
                "unix_time_ns": boundary_timestamp["unix_ns"],
                "monotonic_ns": boundary_monotonic_ns,
                "event_type": (
                    "reward_command_sent"
                    if trial["reward_scheduled"]
                    else (
                        "reward_omission_boundary"
                        if trial["reward_omission_scheduled"]
                        else "nonreward_boundary"
                    )
                ),
                "segment_name": "reward_boundary_1p0_sec",
                "planned_duration_sec": REWARD_DELAY_SEC,
                "segment_boundary_gap_sec": "%.9f" % segment_boundary_gap_sec,
                "command_id": reward_command_id,
                "reward_trigger_source": (
                    "precomputed_open_loop_schedule"
                    if trial["reward_scheduled"]
                    else "no_valve_command"
                ),
                "notes": "reward decision does not depend on licking",
            }
        )
        append_event(event_log_path, boundary_row)

        second_row = make_display_event(
            "stimulus_segment_2",
            trial,
            "second_0p5_sec",
            image_paths["second"],
            POST_REWARD_STIM_SEC,
            second_perf,
            second_timing,
        )
        second_row["segment_boundary_gap_sec"] = "%.9f" % segment_boundary_gap_sec
        append_event(event_log_path, second_row)

        background_row = trial_context(trial, "iti")
        background_row.update(
            {
                "utc_iso": background_timing["request_utc_iso"],
                "unix_time_utc_sec": background_timing["request_unix_sec"],
                "unix_time_ns": background_timing["request_unix_ns"],
                "monotonic_ns": background_timing["request_perf_counter_ns"],
                "event_type": "iti_on",
                "segment_name": "gray_photodiode_off",
                "raw_path": str(background_raw_path),
                "planned_duration_sec": trial["planned_iti_duration_sec"],
                "planned_iti_duration_sec": trial["planned_iti_duration_sec"],
                "display_request_unix_ns": background_timing["request_unix_ns"],
                "display_return_unix_ns": background_timing["return_unix_ns"],
                "display_request_perf_counter_ns": background_timing[
                    "request_perf_counter_ns"
                ],
                "display_return_perf_counter_ns": background_timing[
                    "return_perf_counter_ns"
                ],
                "display_call_duration_sec": "%.9f" % background_timing["duration_sec"],
                "start_time_unix": getattr(background_perf, "start_time", ""),
                "mean_interframe_us": getattr(background_perf, "mean_interframe", ""),
                "stddev_interframe_us": getattr(
                    background_perf, "stddev_interframe", ""
                ),
            }
        )
        append_event(event_log_path, background_row)

        runtime_by_trial[trial["trial_index"]] = {
            "stim_request_unix_ns": first_timing["request_unix_ns"],
            "stim_segment1_return_unix_ns": first_timing["return_unix_ns"],
            "stim_segment2_request_unix_ns": second_timing["request_unix_ns"],
            "stim_offset_request_unix_ns": background_timing["request_unix_ns"],
            "segment_boundary_gap_sec": segment_boundary_gap_sec,
        }

        log_drained_gpio_events(gpio_client, event_log_path, all_gpio_events)
        wait_until(
            iti_deadline,
            gpio_client,
            event_log_path,
            all_gpio_events,
        )
        actual_iti = time.monotonic() - iti_start_monotonic
        append_event(
            event_log_path,
            exact_timestamp_event(
                "iti_end",
                **trial_context(trial, "iti"),
                planned_iti_duration_sec=trial["planned_iti_duration_sec"],
                notes="actual_iti_duration_sec=%.9f" % actual_iti,
            )
        )

        elapsed = time.monotonic() - task_start_monotonic
        average = elapsed / float(completed_count)
        remaining = average * (total_trials - completed_count)
        sys.stdout.write(
            "\rProgress: %d/%d (%.1f%%), estimated task remaining %s   "
            % (
                completed_count,
                total_trials,
                100.0 * completed_count / float(total_trials),
                base.format_seconds(remaining),
            )
        )
        sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()
    return runtime_by_trial


def _count_events_in_window(event_ns, onset_ns, start_sec, stop_sec):
    start_ns = onset_ns + int(round(start_sec * 1_000_000_000))
    stop_ns = onset_ns + int(round(stop_sec * 1_000_000_000))
    return sum(1 for value in event_ns if start_ns <= value < stop_ns)


def build_trial_summary(trials, runtime_by_trial, all_gpio_events):
    lick_ns_by_trial = {}
    reward_on_ns_by_trial = {}
    for event in all_gpio_events:
        trial_index = event.get("trial_index", "")
        if trial_index in ("", None):
            continue
        try:
            trial_index = int(trial_index)
        except (TypeError, ValueError):
            continue
        if event.get("event_type") == "lick_onset":
            lick_ns_by_trial.setdefault(trial_index, []).append(
                int(event["unix_time_ns"])
            )
        elif event.get("event_type") == "reward_valve_on":
            reward_on_ns_by_trial.setdefault(trial_index, []).append(
                int(event["unix_time_ns"])
            )

    rows = []
    for trial in trials:
        trial_index = trial["trial_index"]
        runtime = runtime_by_trial.get(trial_index, {})
        onset_ns = runtime.get("stim_request_unix_ns")
        lick_ns = sorted(lick_ns_by_trial.get(trial_index, []))
        reward_ns = sorted(reward_on_ns_by_trial.get(trial_index, []))
        first_reward_ns = reward_ns[0] if reward_ns else ""
        if onset_ns is None:
            counts = {name: "" for name in (
                "pre", "early", "full", "late", "post", "after"
            )}
            actual_reward_delay = ""
        else:
            counts = {
                "pre": _count_events_in_window(lick_ns, onset_ns, -0.5, 0.0),
                "early": _count_events_in_window(lick_ns, onset_ns, 0.0, 0.5),
                "full": _count_events_in_window(lick_ns, onset_ns, 0.0, 1.0),
                "late": _count_events_in_window(lick_ns, onset_ns, 0.5, 1.0),
                "post": _count_events_in_window(lick_ns, onset_ns, 1.0, 1.5),
                "after": _count_events_in_window(lick_ns, onset_ns, 1.5, 2.0),
            }
            actual_reward_delay = (
                (int(first_reward_ns) - int(onset_ns)) / 1_000_000_000.0
                if first_reward_ns != ""
                else ""
            )
        rows.append(
            {
                "trial_index": trial_index,
                "trial_number": trial["trial_number"],
                "block_number": trial["block_number"],
                "image_role": trial["image_role"],
                "image_category": trial["image_category"],
                "image_id": trial["image_id"],
                "image_filename": trial["image_filename"],
                "reward_eligible": trial["reward_eligible"],
                "reward_scheduled": trial["reward_scheduled"],
                "reward_omission_scheduled": trial["reward_omission_scheduled"],
                "planned_iti_duration_sec": trial["planned_iti_duration_sec"],
                "stim_request_unix_ns": runtime.get("stim_request_unix_ns", ""),
                "stim_segment1_return_unix_ns": runtime.get(
                    "stim_segment1_return_unix_ns", ""
                ),
                "stim_segment2_request_unix_ns": runtime.get(
                    "stim_segment2_request_unix_ns", ""
                ),
                "stim_offset_request_unix_ns": runtime.get(
                    "stim_offset_request_unix_ns", ""
                ),
                "segment_boundary_gap_sec": runtime.get(
                    "segment_boundary_gap_sec", ""
                ),
                "first_reward_valve_on_unix_ns": first_reward_ns,
                "actual_reward_delay_from_software_stim_request_sec": actual_reward_delay,
                "lick_count_pre_0p5_sec": counts["pre"],
                "lick_count_0_to_0p5_sec": counts["early"],
                "lick_count_0_to_1p0_sec": counts["full"],
                "lick_count_0p5_to_1p0_sec": counts["late"],
                "lick_count_1p0_to_1p5_sec": counts["post"],
                "lick_count_1p5_to_2p0_sec": counts["after"],
                "notes": (
                    "Software-aligned summary; use photodiode/DAQ timing for final neural analysis."
                ),
            }
        )
    return rows


def write_rows(path, rows, fields):
    base.write_csv(path, rows, fields)


def estimate_task_seconds(trials):
    return sum(
        STIM_DURATION_SEC + float(trial["planned_iti_duration_sec"])
        for trial in trials
    )


def maybe_import_camera_support(no_camera):
    if no_camera:
        return None
    try:
        import run_stringer_vstim_cam as camera_support

        return camera_support
    except Exception as exc:
        print("Remote camera support unavailable: %s" % exc)
        return None


def main(argv=None):
    args = parse_args(argv)
    base.print_environment()
    try:
        import rpg
    except ImportError as exc:
        raise RuntimeError(
            "The rpg package is not installed. Install the SjulsonLab rpg repo "
            "on the behavior Pi first."
        ) from exc

    hardware_config = load_hardware_config(
        args.hardware_config, simulate_gpio=args.simulate_gpio
    )
    reward_train_duration_sec = (
        hardware_config["reward_num_pulses"]
        * hardware_config["reward_pulse_on_sec"]
        + max(0, hardware_config["reward_num_pulses"] - 1)
        * hardware_config["reward_pulse_off_sec"]
    )
    camera_support = maybe_import_camera_support(args.no_camera)

    mouse_id_raw = base.prompt_text("Mouse ID: ")
    mouse_id = base.sanitize_text(mouse_id_raw) or "mouse"
    session_notes = base.prompt_text("Session notes, optional: ").strip()
    n_blocks = base.prompt_int_or_default(
        "Number of 50-trial probability blocks", DEFAULT_N_BLOCKS
    )
    iti_min_sec = prompt_float_or_default(
        "Minimum gray ITI in seconds", DEFAULT_ITI_MIN_SEC, minimum=0.1
    )
    iti_max_sec = prompt_float_or_default(
        "Maximum gray ITI in seconds", DEFAULT_ITI_MAX_SEC, minimum=iti_min_sec
    )
    pre_background_min = prompt_float_or_default(
        "PRE gray-background duration in minutes",
        DEFAULT_PRE_BACKGROUND_MIN,
        minimum=0.0,
    )
    post_background_min = prompt_float_or_default(
        "POST gray-background duration in minutes",
        DEFAULT_POST_BACKGROUND_MIN,
        minimum=0.0,
    )
    use_camera = False
    if camera_support is not None:
        use_camera = base.prompt_yes_no("Record the remote face camera", default_yes=True)

    image_dir = base.resolve_image_dir()
    all_pngs = base.list_png_files(image_dir)
    assignment_rows, assignment_path, assignment_created, assignment_seed = (
        create_or_load_assignment(
            mouse_id,
            all_pngs,
            ASSIGNMENT_DIR,
        )
    )
    trials, sequence_seed = make_trial_plan(
        assignment_rows,
        n_blocks=n_blocks,
        iti_min_sec=iti_min_sec,
        iti_max_sec=iti_max_sec,
        mouse_id=mouse_id,
        stim_duration_sec=STIM_DURATION_SEC,
        reward_delay_sec=REWARD_DELAY_SEC,
    )
    plan_summary = summarize_trial_plan(trials)
    estimated_task_sec = estimate_task_seconds(trials)
    estimated_total_sec = (
        pre_background_min * 60.0
        + estimated_task_sec
        + post_background_min * 60.0
    )

    print()
    print("Session setup summary:")
    print("  Mouse: %s" % mouse_id)
    print("  Shared 14-image panel: %s" % global_panel_path(ASSIGNMENT_DIR))
    print("  Fixed per-mouse role assignment: %s" % assignment_path)
    print("  Assignment newly created: %s" % assignment_created)
    print("  Blocks: %d x %d trials" % (n_blocks, TRIALS_PER_BLOCK))
    print("  Total trials: %d" % len(trials))
    print("  Stimulus: 1.5 s; open-loop reward boundary: 1.0 s")
    print("  Reward probability: %.3f (fixed)" % REWARD_PROBABILITY)
    print("  ITI: uniform %.3f-%.3f s" % (iti_min_sec, iti_max_sec))
    print("  PRE gray background: %s" % base.format_seconds(pre_background_min * 60.0))
    print("  POST gray background: %s" % base.format_seconds(post_background_min * 60.0))
    print("  Estimated PRE + task + POST: %s" % base.format_seconds(estimated_total_sec))
    print("  Reward pin: BCM%d" % hardware_config["reward_pin_bcm"])
    print("  Lick pin: BCM%d" % hardware_config["lick_pin_bcm"])
    print("  Reward pulse-train duration: %.3f s" % reward_train_duration_sec)
    if reward_train_duration_sec > POST_REWARD_STIM_SEC:
        print(
            "  WARNING: reward train extends beyond the 0.5 s post-boundary "
            "image segment and into the ITI."
        )
    print("  GPIO simulation: %s" % hardware_config["simulate_gpio"])
    print("  Face camera: %s" % use_camera)
    print("  IMPORTANT: reward delivery is independent of licking.")
    print()
    for row in plan_summary:
        print(
            "  %-22s  n=%3d  reward=%3d  omission=%3d  %s"
            % (
                row["image_role"],
                row["n_presentations"],
                row["n_rewards"],
                row["n_omissions"],
                row["image_filename"],
            )
        )
    if not base.prompt_yes_no("Create files and prepare this session", default_yes=True):
        print("Session aborted before hardware start.")
        return 0

    session_stamp = base.utc_session_stamp()
    session_id = make_session_name(mouse_id, session_stamp)
    session_root = base.ensure_dir(OUTPUT_ROOT / session_id)
    raw_cache_root = base.ensure_dir(session_root / "raw_cache")
    event_log_path = session_root / (session_id + "_event_log.csv")
    camera_event_log_path = session_root / (session_id + "_camera_event_log.csv")
    planned_sequence_path = session_root / (session_id + "_planned_sequence.csv")
    selected_images_path = session_root / (session_id + "_image_assignment.csv")
    plan_summary_path = session_root / (session_id + "_plan_summary.csv")
    trial_summary_path = session_root / (session_id + "_trial_summary.csv")
    metadata_path = session_root / (session_id + "_metadata.json")

    write_rows(selected_images_path, assignment_rows, [
        "image_role",
        "image_category",
        "presentation_probability",
        "reward_eligible",
        "image_id",
        "image_filename",
        "image_path",
    ])
    write_rows(planned_sequence_path, trials, PLANNED_SEQUENCE_FIELDS)
    write_rows(plan_summary_path, plan_summary, [
        "image_role",
        "image_filename",
        "n_presentations",
        "n_rewards",
        "n_omissions",
        "realized_reward_probability",
    ])

    metadata = {
        "session_id": session_id,
        "utc_iso_created": base.utc_iso_now(),
        "mouse_id_input": mouse_id_raw,
        "mouse_id": mouse_id,
        "session_notes": session_notes,
        "protocol": "open_loop_natural_image_reward_conditioning",
        "reward_is_lick_contingent": False,
        "reward_trigger_rule": "precomputed schedule only; licking is recorded but never gates reward",
        "n_unique_images": 14,
        "n_blocks": n_blocks,
        "trials_per_block": TRIALS_PER_BLOCK,
        "total_trials": len(trials),
        "reward_probability": REWARD_PROBABILITY,
        "reward_probability_fixed": True,
        "stim_duration_sec": STIM_DURATION_SEC,
        "reward_delay_sec": REWARD_DELAY_SEC,
        "post_reward_stim_sec": POST_REWARD_STIM_SEC,
        "iti_distribution": "uniform",
        "iti_min_sec": iti_min_sec,
        "iti_max_sec": iti_max_sec,
        "pre_background_requested_sec": pre_background_min * 60.0,
        "post_background_requested_sec": post_background_min * 60.0,
        "background_visual_condition": "gray_127_with_black_photodiode_patch",
        "global_image_panel_path": str(global_panel_path(ASSIGNMENT_DIR)),
        "assignment_path": str(assignment_path),
        "assignment_created_this_session": assignment_created,
        "resolved_assignment_seed": assignment_seed,
        "resolved_sequence_seed": sequence_seed,
        "image_dir": str(image_dir),
        "hardware_config_path": str(Path(args.hardware_config)),
        "hardware_config": hardware_config,
        "reward_train_duration_sec": reward_train_duration_sec,
        "reward_train_extends_into_iti": reward_train_duration_sec > POST_REWARD_STIM_SEC,
        "base_ttl_bcm23_used": False,
        "face_camera_requested": use_camera,
        "screen_resolution": list(base.SCREEN_RESOLUTION),
        "screen_background_gray": base.SCREEN_BACKGROUND_GRAY,
        "refresh_rate_hz": base.REFRESH_RATE_HZ,
        "photodiode_patch_enabled": base.ENABLE_PHOTODIODE_PATCH,
        "photodiode_size_px": base.PHOTODIODE_SIZE_PX,
        "vstim_natural_git_commit": get_git_commit(),
        "planned_sequence_csv": str(planned_sequence_path),
        "event_log_csv": str(event_log_path),
        "trial_summary_csv": str(trial_summary_path),
        "camera_event_log_csv": str(camera_event_log_path) if use_camera else "",
        "plan_summary": plan_summary,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    all_gpio_events = []
    runtime_by_trial = {}
    gpio_client = BehaviorGPIOClient(hardware_config)
    camera_started = False
    camera_stopped = False
    camera_fetch_completed = False
    camera_conversion_completed = False
    session_completed = False

    try:
        print("Building one-frame gray background raw...")
        background_raw_path = build_background_raw(rpg, raw_cache_root)
        print("Starting independent reward/lick GPIO process...")
        ready_state = gpio_client.start()
        append_event(
            event_log_path,
            exact_timestamp_event(
                "gpio_parent_confirmed_ready",
                phase="startup",
                reward_pin_bcm=ready_state["reward_pin_bcm"],
                lick_pin_bcm=ready_state["lick_pin_bcm"],
                notes="worker_pid=%s; simulate_gpio=%s"
                % (ready_state["worker_pid"], ready_state["simulate_gpio"]),
            ),
        )
        log_drained_gpio_events(gpio_client, event_log_path, all_gpio_events)

        with rpg.Screen(
            base.SCREEN_RESOLUTION,
            background=base.SCREEN_BACKGROUND_GRAY,
            colormode=base.SCREEN_COLORMODE,
        ) as screen:
            loaded_background_raw = screen.load_raw(str(background_raw_path))
            screen.display_raw(loaded_background_raw)
            gpio_client.set_context(background_context("preparation_gray"))
            append_event(
                event_log_path,
                exact_timestamp_event(
                    "preparation_gray_on",
                    phase="preparation_gray",
                    raw_path=str(background_raw_path),
                    notes="gray visible before stimulus raw conversion",
                ),
            )

            print("Building 14 image raws while the screen remains gray...")
            raw_build_start = time.monotonic()
            raw_paths = build_stimulus_segment_raws(
                rpg, raw_cache_root, assignment_rows
            )
            raw_build_sec = time.monotonic() - raw_build_start
            loaded_raws = {}
            for image_filename, paths in raw_paths.items():
                loaded_raws[image_filename] = {
                    "first": screen.load_raw(str(paths["first"])),
                    "second": screen.load_raw(str(paths["second"])),
                }
            metadata["raw_cache_build_duration_sec"] = raw_build_sec
            append_event(
                event_log_path,
                exact_timestamp_event(
                    "stimulus_raw_cache_ready",
                    phase="preparation_gray",
                    notes="build_duration_sec=%.9f" % raw_build_sec,
                ),
            )

            if base.prompt_yes_no(
                "Deliver one manual test reward before starting baselines",
                default_yes=False,
            ):
                command_id = gpio_client.manual_reward()
                append_event(
                    event_log_path,
                    exact_timestamp_event(
                        "manual_reward_command_sent",
                        phase="manual_reward",
                        command_id=command_id,
                        notes="operator_requested_test_reward",
                    ),
                )
                manual_duration = (
                    hardware_config["reward_num_pulses"]
                    * hardware_config["reward_pulse_on_sec"]
                    + max(0, hardware_config["reward_num_pulses"] - 1)
                    * hardware_config["reward_pulse_off_sec"]
                    + 0.25
                )
                wait_until(
                    time.monotonic() + manual_duration,
                    gpio_client,
                    event_log_path,
                    all_gpio_events,
                )

            if use_camera:
                append_event(
                    event_log_path,
                    exact_timestamp_event(
                        "camera_start_requested",
                        phase="preparation_gray",
                        notes="screen_gray=true",
                    ),
                )
                camera_result = camera_support.start_camera_recording_with_recovery(
                    mouse_id,
                    session_id,
                    camera_event_log_path,
                )
                if not camera_result["confirmed_running"]:
                    raise RuntimeError(
                        "Remote camera was not confirmed: %s"
                        % camera_result.get("error", "unknown error")
                    )
                camera_started = True
                metadata["camera_start_result"] = camera_result
                append_event(
                    event_log_path,
                    exact_timestamp_event(
                        "camera_recording_confirmed",
                        phase="preparation_gray",
                        notes="remote_camera_output_growing_confirmed",
                    ),
                )

            base.prompt_text(
                "Start the 2P acquisition now, then press Enter to begin the PRE gray background: "
            )
            append_event(
                event_log_path,
                exact_timestamp_event(
                    "two_photon_operator_gate_released",
                    phase="preparation_gray",
                    notes="operator_pressed_enter",
                ),
            )

            pre_actual = hold_background(
                "prestim_background",
                pre_background_min * 60.0,
                gpio_client,
                event_log_path,
                all_gpio_events,
            )
            metadata["pre_background_actual_sec"] = pre_actual

            append_event(
                event_log_path,
                exact_timestamp_event(
                    "task_start",
                    phase="task",
                    notes="first stimulus follows; reward is open-loop",
                ),
            )
            runtime_by_trial = run_trials(
                screen,
                trials,
                loaded_raws,
                loaded_background_raw,
                raw_paths,
                background_raw_path,
                gpio_client,
                event_log_path,
                all_gpio_events,
            )
            append_event(
                event_log_path,
                exact_timestamp_event(
                    "task_end",
                    phase="task",
                    notes="all planned trials completed",
                ),
            )

            post_actual = hold_background(
                "poststim_background",
                post_background_min * 60.0,
                gpio_client,
                event_log_path,
                all_gpio_events,
            )
            metadata["post_background_actual_sec"] = post_actual
            session_completed = True

            # Keep exactly the same gray background on screen during remote
            # camera cleanup.  The current natural-stim camera runner switches
            # to black; this task intentionally does not.
            if use_camera and camera_started:
                try:
                    camera_support.stop_camera_recording()
                    camera_stopped = True
                    append_event(
                        event_log_path,
                        exact_timestamp_event(
                            "camera_stop_confirmed",
                            phase="poststim_camera_cleanup_gray",
                            notes="screen_remained_gray",
                        ),
                    )
                    fetch_state = camera_support.fetch_camera_recording()
                    camera_fetch_completed = bool(
                        fetch_state.get("camera_fetch_completed", True)
                    )
                    append_event(
                        event_log_path,
                        exact_timestamp_event(
                            "camera_fetch_returned",
                            phase="poststim_camera_cleanup_gray",
                            notes=json.dumps(fetch_state, sort_keys=True),
                        ),
                    )
                    convert_state = camera_support.convert_camera_recording()
                    camera_conversion_completed = bool(
                        convert_state.get("camera_conversion_completed", True)
                    )
                    append_event(
                        event_log_path,
                        exact_timestamp_event(
                            "camera_conversion_returned",
                            phase="poststim_camera_cleanup_gray",
                            notes=json.dumps(convert_state, sort_keys=True),
                        ),
                    )
                except Exception as exc:
                    append_event(
                        event_log_path,
                        exact_timestamp_event(
                            "camera_cleanup_error",
                            phase="poststim_camera_cleanup_gray",
                            notes=str(exc),
                        ),
                    )
                    print("Camera cleanup error: %s" % exc, file=sys.stderr)

    except KeyboardInterrupt:
        append_event(
            event_log_path,
            exact_timestamp_event(
                "session_interrupted",
                phase="unknown",
                notes="KeyboardInterrupt",
            ),
        )
        raise
    except Exception as exc:
        append_event(
            event_log_path,
            exact_timestamp_event(
                "session_error",
                phase="unknown",
                notes="%s: %s" % (type(exc).__name__, exc),
            ),
        )
        raise
    finally:
        if camera_started and not camera_stopped and camera_support is not None:
            try:
                camera_support.stop_camera_recording()
                camera_stopped = True
            except Exception as exc:
                print("Emergency camera stop failed: %s" % exc, file=sys.stderr)

        if gpio_client is not None:
            try:
                shutdown_events = gpio_client.shutdown()
                for event in shutdown_events:
                    append_event(event_log_path, event)
                    if event.get("message_type") == "event":
                        all_gpio_events.append(dict(event))
            except Exception as exc:
                print("GPIO shutdown error: %s" % exc, file=sys.stderr)

        trial_summary = build_trial_summary(
            trials, runtime_by_trial, all_gpio_events
        )
        write_rows(trial_summary_path, trial_summary, TRIAL_SUMMARY_FIELDS)

        metadata["utc_iso_end"] = base.utc_iso_now()
        metadata["session_completed"] = session_completed
        metadata["camera_started"] = camera_started
        metadata["camera_stopped"] = camera_stopped
        metadata["camera_fetch_completed"] = camera_fetch_completed
        metadata["camera_conversion_completed"] = camera_conversion_completed
        metadata["n_gpio_events_logged"] = len(all_gpio_events)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

        if session_completed:
            print("Session finished. Files are in: %s" % session_root)
        else:
            print("Session stopped early. Partial files are in: %s" % session_root)
        print("Trial summary: %s" % trial_summary_path)
        print(
            "For neural alignment, use the recorded photodiode/DAQ signal as "
            "ground truth; software timestamps are retained for diagnostics."
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise
