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
import math
import os
import statistics
import subprocess
import sys
import time
import select
from pathlib import Path

import run_stringer_vstim as base

from reward_conditioning_gpio import BehaviorGPIOClient
from remote_camera_control import CAMERA_EVENT_LOG_FILENAME
from rig_telemetry import (
    DEFAULT_HOST as DEFAULT_TELEMETRY_HOST,
    DEFAULT_PORT as DEFAULT_TELEMETRY_PORT,
    TelemetryPublisher,
)
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
DEFAULT_ITI_MIN_SEC = 3.0
DEFAULT_ITI_MAX_SEC = 4.5
MINIMUM_CLEAN_POST_SUCTION_SEC = 0.75
DEFAULT_PRE_BACKGROUND_MIN = 5.0
DEFAULT_POST_BACKGROUND_MIN = 5.0
BACKGROUND_POLL_SEC = 0.10
STATUS_TTY_INTERVAL_SEC = 1.0
STATUS_NON_TTY_INTERVAL_SEC = 30.0
REWARD_VERIFICATION_MARGIN_SEC = 0.25
QC_SCHEMA_VERSION = 1
SESSION_OUTPUT_SCHEMA_VERSION = 2
EVENT_LOG_SCHEMA_VERSION = 1
TRIAL_SUMMARY_SCHEMA_VERSION = 2

CORE_OPERATIONAL_TIMING_FIELDS = (
    "operator_gate_enter_monotonic_ns",
    "operator_gate_release_monotonic_ns",
    "operator_gate_wait_sec",
    "pre_start_monotonic_ns",
    "pre_end_monotonic_ns",
    "pre_elapsed_sec",
    "task_start_monotonic_ns",
    "task_end_monotonic_ns",
    "post_start_monotonic_ns",
    "post_end_monotonic_ns",
    "post_elapsed_sec",
)
CAMERA_OPERATIONAL_TIMING_FIELDS = (
    "camera_recording_request_monotonic_ns",
    "camera_recording_confirmed_monotonic_ns",
    "camera_stop_confirmed_monotonic_ns",
    "camera_recording_elapsed_local_sec",
)

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
    "suction_scheduled",
    "planned_suction_delay_sec",
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
    "suction_pin_bcm",
    "lick_edge",
    "reward_pulse_on_sec",
    "reward_pulse_off_sec",
    "reward_pulse_index",
    "reward_num_pulses",
    "reward_trigger_source",
    "suction_duration_sec",
    "suction_trigger_source",
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
    "suction_scheduled",
    "trial_executed",
    "stim_presented",
    "trial_completed",
    "planned_iti_duration_sec",
    "stim_request_unix_ns",
    "stim_request_monotonic_ns",
    "stim_segment1_return_unix_ns",
    "stim_segment2_request_unix_ns",
    "stim_offset_request_unix_ns",
    "stim_offset_monotonic_ns",
    "segment_boundary_gap_sec",
    "reward_command_id",
    "suction_command_id",
    "suction_command_unix_ns",
    "suction_command_monotonic_ns",
    "suction_on_unix_ns",
    "suction_on_monotonic_ns",
    "suction_off_unix_ns",
    "suction_off_monotonic_ns",
    "suction_complete_unix_ns",
    "suction_complete_monotonic_ns",
    "software_suction_delay_sec",
    "estimated_suction_delay_from_physical_onset_sec",
    "first_reward_valve_on_unix_ns",
    "actual_reward_delay_from_software_stim_request_sec",
    "lick_count_pre_0p5_sec",
    "lick_count_0_to_0p5_sec",
    "lick_count_0_to_1p0_sec",
    "lick_count_0p5_to_1p0_sec",
    "lick_count_1p0_to_1p5_sec",
    "lick_count_1p5_to_2p0_sec",
    "lick_count_2p0_to_3p0_sec",
    "lick_count_3p0_to_3p5_sec",
    "lick_count_3p5_to_4p0_sec",
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
    parser.add_argument("--mouse-id")
    parser.add_argument("--session-notes")
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--iti-min-sec", type=float)
    parser.add_argument("--iti-max-sec", type=float)
    parser.add_argument("--pre-background-min", type=float)
    parser.add_argument("--post-background-min", type=float)
    camera = parser.add_mutually_exclusive_group()
    camera.add_argument("--camera", action="store_true")
    camera.add_argument("--no-camera", action="store_true")
    parser.add_argument(
        "--telemetry-host",
        default=os.environ.get("RIG_MONITOR_HOST", DEFAULT_TELEMETRY_HOST),
        help="Read-only UDP telemetry monitor host (default: 192.168.1.150).",
    )
    parser.add_argument(
        "--telemetry-port",
        type=int,
        default=int(os.environ.get("RIG_MONITOR_PORT", DEFAULT_TELEMETRY_PORT)),
        help="Read-only UDP telemetry monitor port (default: 5055).",
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Disable optional read-only UDP telemetry.",
    )
    return parser.parse_args(argv)


def _strict_prompt(prompt):
    try:
        value = input(prompt)
    except EOFError as exc:
        raise RuntimeError("Unexpected EOF while waiting for operator input.") from exc
    return value.strip()


def prompt_int_with_default(prompt, default_value, minimum=None):
    while True:
        raw = _strict_prompt("%s [%s]: " % (prompt, default_value))
        try:
            value = int(default_value) if not raw else int(raw)
            if minimum is not None and value < minimum: raise ValueError("at least %s" % minimum)
            return value
        except ValueError as exc:
            print("Invalid integer (%s); try again." % exc)


def prompt_yes_no_strict(prompt, default_yes=None):
    suffix = " [Y/n]" if default_yes is True else " [y/N]" if default_yes is False else " [y/n]"
    while True:
        raw = _strict_prompt(prompt + suffix + ": ").lower()
        if not raw and default_yes is not None: return bool(default_yes)
        if raw in ("y", "yes"): return True
        if raw in ("n", "no"): return False
        print("Please answer y/yes or n/no.")


def _predicted_finish_text(remaining_sec, wall_time_sec=None):
    finish_epoch = (time.time() if wall_time_sec is None else float(wall_time_sec)) + max(0.0, float(remaining_sec or 0.0))
    return time.strftime("%H:%M:%S", time.localtime(finish_epoch))


def format_operator_status(phase, camera_elapsed_sec=None, remaining_sec=None,
                           trial_number=None, total_trials=None, post_sec=None,
                           protocol_remaining_sec=None, wall_time_sec=None):
    prefix = "CAM OFF" if camera_elapsed_sec is None else "REC %s" % base.format_seconds(camera_elapsed_sec)
    if phase == "WAITING_FOR_2P": return "%s | WAITING FOR 2P | press Enter when acquisition is running | planned after Enter %s" % (prefix, base.format_seconds(remaining_sec or 0))
    if phase == "PRE":
        protocol_remaining = remaining_sec if protocol_remaining_sec is None else protocol_remaining_sec
        return "%s | PRE %s remaining | protocol remaining %s | finish ~%s" % (
            prefix, base.format_seconds(remaining_sec or 0),
            base.format_seconds(protocol_remaining or 0),
            _predicted_finish_text(protocol_remaining, wall_time_sec))
    if phase == "TASK":
        percent = 0.0 if not total_trials else 100.0 * float(trial_number) / float(total_trials)
        finish_remaining = float(remaining_sec or 0.0) + float(post_sec or 0.0)
        return "%s | TASK %d/%d (%.1f%%) | planned task remaining %s | +POST %s | finish ~%s" % (
            prefix, trial_number, total_trials, percent,
            base.format_seconds(remaining_sec or 0), base.format_seconds(post_sec or 0),
            _predicted_finish_text(finish_remaining, wall_time_sec))
    if phase == "POST":
        return "%s | POST %s remaining | finish ~%s" % (
            prefix, base.format_seconds(remaining_sec or 0),
            _predicted_finish_text(remaining_sec, wall_time_sec))
    return "%s | %s %s remaining" % (prefix, phase, base.format_seconds(remaining_sec or 0))


class StatusReporter(object):
    """Rate-limited terminal-only operational status; never writes event rows."""
    def __init__(self, stream=None, monotonic_fn=None):
        self.stream = stream or sys.stdout
        self.monotonic_fn = monotonic_fn or time.monotonic
        self.last_print = None

    def report(self, text, force=False):
        now = self.monotonic_fn()
        interval = STATUS_TTY_INTERVAL_SEC if self.stream.isatty() else STATUS_NON_TTY_INTERVAL_SEC
        if not force and self.last_print is not None and now - self.last_print < interval:
            return False
        self.stream.write(("\r" if self.stream.isatty() else "") + text + ("" if self.stream.isatty() else "\n"))
        self.stream.flush()
        self.last_print = now
        return True

    def finalize(self, text=None):
        if text: self.report(text, force=True)
        if self.stream.isatty(): self.stream.write("\n"); self.stream.flush()


class TelemetryStateReporter(object):
    """One-second state heartbeat layered onto existing wait callbacks."""

    def __init__(self, publisher, interval_sec=1.0, monotonic_fn=None):
        self.publisher = publisher
        self.interval_sec = float(interval_sec)
        self.monotonic_fn = monotonic_fn or time.monotonic
        self.last_report = None
        self.parent_error_count = 0

    def report_state(self, payload, force=False):
        try:
            if self.publisher is None or not self.publisher.enabled:
                return False
            now = self.monotonic_fn()
            if (not force and self.last_report is not None
                    and now - self.last_report < self.interval_sec):
                return False
            published = self.publisher.publish_state(payload)
            self.last_report = now
            return published
        except Exception:
            _record_telemetry_parent_error(self)
            return False


def _record_telemetry_parent_error(*objects):
    for obj in objects:
        if obj is None:
            continue
        try:
            obj.parent_error_count = getattr(obj, "parent_error_count", 0) + 1
        except Exception:
            pass


def safe_report_telemetry_state(reporter, phase, trial=None,
                                total_trials=0, total_blocks=0, eta_sec=None,
                                water_accounting=None, force=False):
    """Build and enqueue telemetry state without affecting experiment flow."""
    if reporter is None:
        return False
    try:
        return reporter.report_state(
            build_telemetry_state_payload(
                phase, trial, total_trials, total_blocks, eta_sec,
                water_accounting),
            force=force,
        )
    except Exception:
        _record_telemetry_parent_error(
            reporter, getattr(reporter, "publisher", None))
        return False


def derive_session_status(task_completed, post_completed, interrupted=False, primary_error=None,
                          camera_enabled=False, camera_state=None, cleanup_errors=None,
                          finalization_errors=None, camera_cleanup_error=False,
                          camera_data_secured=None):
    camera_state = camera_state or {}
    if primary_error is not None: return "failed"
    if interrupted: return "interrupted"
    if not (task_completed and post_completed): return "incomplete"
    if (camera_cleanup_error
            or camera_state.get("camera_cleanup_error")
            or camera_state.get("remote_raw_cleanup_error")):
        return "protocol_complete_camera_cleanup_failed"
    if cleanup_errors or finalization_errors: return "cleanup_failed"
    if not camera_enabled: return "complete"
    if camera_data_secured is None:
        camera_data_secured = derive_camera_data_secured(
            camera_enabled, camera_state, camera_cleanup_error)
    if camera_data_secured:
        return "complete"
    if (camera_state.get("camera_stop_confirmed")
            and camera_state.get("camera_raw_files_verified")
            and camera_state.get("camera_raw_hash_verified")
            and not camera_state.get("remote_raw_cleanup_error")
            and not camera_state.get("camera_cleanup_error")):
        return "protocol_complete_video_pending"
    return "protocol_complete_camera_cleanup_failed"


def wait_for_two_photon_gate(status_callback=None, service_callback=None, poll_sec=0.25):
    """Wait for an explicit Enter; EOF never releases the experimental gate."""
    if not sys.stdin.isatty():
        raise RuntimeError("2P operator gate requires interactive stdin; refusing to auto-bypass.")
    started = time.monotonic()
    while True:
        if service_callback: service_callback()
        if status_callback: status_callback()
        ready, _, _ = select.select([sys.stdin], [], [], poll_sec)
        if not ready: continue
        line = sys.stdin.readline()
        if line == "": raise RuntimeError("EOF received while waiting for 2P operator gate.")
        if line.rstrip("\r\n") == "": return time.monotonic() - started
        print("Press Enter to begin PRE.")


def prompt_float_or_default(prompt, default_value, minimum=None, maximum=None):
    while True:
        raw = _strict_prompt("%s [%s]: " % (prompt, default_value))
        try:
            value = float(default_value) if not raw else float(raw)
            if not math.isfinite(value): raise ValueError("finite")
            if minimum is not None and value < minimum: raise ValueError("at least %s" % minimum)
            if maximum is not None and value > maximum: raise ValueError("at most %s" % maximum)
            return value
        except ValueError as exc:
            print("Invalid value (%s); try again." % exc)


def planned_task_remaining_seconds(trials, next_trial_index):
    return estimate_task_seconds(trials[next_trial_index:]) if next_trial_index < len(trials) else 0.0


def planned_task_remaining_during_iti(trials, current_trial_index, iti_deadline_monotonic,
                                      now_monotonic=None):
    """Exact task remainder while the current non-final trial is in its ITI."""
    now = time.monotonic() if now_monotonic is None else float(now_monotonic)
    current_iti_remaining = max(0.0, float(iti_deadline_monotonic) - now)
    return current_iti_remaining + planned_task_remaining_seconds(
        trials, int(current_trial_index) + 1)


def atomic_write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(".%s.tmp" % path.name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists(): temporary.unlink()


def make_session_name(mouse_id, session_stamp):
    return "%s_%s_%s" % (mouse_id, session_stamp, SESSION_SUFFIX)


def final_camera_metadata(latest_state, use_camera, camera_started=False,
                          camera_stop_confirmed=False, camera_fetch_completed=False,
                          camera_conversion_completed=False):
    """Summarize the most recent camera controller state for session metadata."""
    latest_state = dict(latest_state or {})
    return {
        "camera_started": bool(latest_state.get("camera_output_growing_confirmed", camera_started)),
        "camera_stop_confirmed": bool(latest_state.get("camera_stop_confirmed", camera_stop_confirmed)),
        "camera_fetch_completed": bool(latest_state.get("camera_fetch_completed", camera_fetch_completed)),
        "camera_conversion_completed": bool(latest_state.get("camera_conversion_completed", camera_conversion_completed)),
        "camera_transfer_completed": bool(latest_state.get("camera_transfer_completed", latest_state.get("camera_fetch_completed", camera_fetch_completed))),
        "camera_raw_files_verified": bool(latest_state.get("camera_raw_files_verified", False)),
        "camera_raw_hash_verified": bool(latest_state.get("camera_raw_hash_verified", False)),
        "camera_mp4_verified": bool(latest_state.get("camera_mp4_verified", False)),
        "remote_raw_cleanup_completed": bool(latest_state.get("remote_raw_cleanup_completed", False)),
        "remote_raw_retained": bool(latest_state.get("remote_raw_retained", use_camera)),
        "remote_raw_cleanup_error": latest_state.get("remote_raw_cleanup_error", ""),
    }


def derive_camera_data_secured(camera_enabled, camera_state=None,
                               camera_cleanup_error=False):
    """Return the single archival-success decision used by session outputs."""
    if not camera_enabled:
        return True
    state = dict(camera_state or {})
    required = (
        "camera_stop_confirmed",
        "camera_raw_files_verified",
        "camera_raw_hash_verified",
        "camera_mp4_verified",
        "remote_raw_cleanup_completed",
    )
    return bool(
        not camera_cleanup_error
        and not state.get("camera_cleanup_error")
        and not state.get("remote_raw_cleanup_error")
        and all(state.get(field) is True for field in required)
    )


def resolve_final_completion_state(
        task_completed, post_background_completed, interrupted=False,
        primary_error=None, camera_enabled=False, camera_state=None,
        camera_cleanup_error=False, cleanup_errors=None,
        finalization_errors=None):
    camera_data_secured = derive_camera_data_secured(
        camera_enabled, camera_state, camera_cleanup_error)
    session_status = derive_session_status(
        task_completed, post_background_completed, interrupted, primary_error,
        camera_enabled, camera_state, cleanup_errors, finalization_errors,
        camera_cleanup_error=camera_cleanup_error,
        camera_data_secured=camera_data_secured)
    return {
        "camera_data_secured": camera_data_secured,
        "session_completed": session_status == "complete",
        "session_status": session_status,
    }


def final_session_exit(primary_error, interrupted, cleanup_errors, finalization_errors,
                       camera_cleanup_error=False, camera_cleanup_error_message=""):
    """Return the final exit code or re-raise the authoritative failure."""
    if primary_error is not None:
        raise primary_error
    if interrupted:
        return 130
    if camera_cleanup_error:
        detail = camera_cleanup_error_message or "explicit camera cleanup exception"
        raise RuntimeError("Camera cleanup failed: %s" % detail)
    if cleanup_errors or finalization_errors:
        raise RuntimeError("Session cleanup/finalization failed: %s" % "; ".join(cleanup_errors + finalization_errors))
    return 0


def camera_elapsed_from_ns(recording_confirmed_ns, stop_confirmed_ns):
    if recording_confirmed_ns is None or stop_confirmed_ns is None:
        return None
    return max(0.0, (int(stop_confirmed_ns) - int(recording_confirmed_ns)) / 1e9)


def camera_startup_confirmation_latency_from_ns(
        recording_request_ns, recording_confirmed_ns):
    if recording_request_ns is None or recording_confirmed_ns is None:
        return None
    return max(0.0, (int(recording_confirmed_ns) - int(recording_request_ns)) / 1e9)


def resolve_camera_choice(camera_requested, no_camera_requested, camera_support,
                          prompt_fn=prompt_yes_no_strict):
    if camera_requested:
        if camera_support is None:
            raise RuntimeError("--camera was requested but camera support is unavailable.")
        return True
    if no_camera_requested or camera_support is None:
        return False
    return bool(prompt_fn("Record face camera", default_yes=True))


def update_session_manifest_status(manifest_path, status):
    manifest = json.loads(Path(manifest_path).read_text())
    manifest["status"] = status
    atomic_write_json(manifest_path, manifest)
    return manifest


def _manifest_artifact_reference(session_root, path):
    if path is None:
        return {"path": None, "exists": False}
    path = Path(path)
    try:
        relative = str(path.relative_to(Path(session_root)))
    except ValueError:
        relative = str(path)
    return {"path": relative, "exists": bool(path.is_file())}


def build_session_manifest(session_root, session_id, mouse_id, protocol,
                           status, artifact_paths):
    return {
        "session_id": session_id,
        "mouse_id": mouse_id,
        "protocol": protocol,
        "session_output_schema_version": SESSION_OUTPUT_SCHEMA_VERSION,
        "status": status,
        "files": {
            name: _manifest_artifact_reference(session_root, path)
            for name, path in artifact_paths.items()
        },
    }


def finalize_status_artifacts(
        metadata, metadata_path, manifest_path, session_root, session_id,
        mouse_id, protocol, artifact_paths, completion_inputs,
        cleanup_errors, finalization_errors, write_json_fn=None):
    """Write final metadata/manifest with one bounded consistency repair pass."""
    writer = write_json_fn or atomic_write_json

    def resolve_and_apply():
        inputs = dict(completion_inputs)
        inputs["cleanup_errors"] = cleanup_errors
        inputs["finalization_errors"] = finalization_errors
        state = resolve_final_completion_state(**inputs)
        metadata.update(state)
        metadata["cleanup_errors"] = list(cleanup_errors)
        metadata["finalization_errors"] = list(finalization_errors)
        return state

    def manifest_payload(state):
        return build_session_manifest(
            session_root, session_id, mouse_id, protocol,
            state["session_status"], artifact_paths)

    state = resolve_and_apply()
    metadata_ok = False
    manifest_ok = False
    try:
        writer(metadata_path, metadata)
        metadata_ok = True
    except Exception as exc:
        finalization_errors.append(
            "metadata: %s: %s" % (type(exc).__name__, exc))
        state = resolve_and_apply()

    try:
        writer(manifest_path, manifest_payload(state))
        manifest_ok = True
    except Exception as exc:
        finalization_errors.append(
            "manifest: %s: %s" % (type(exc).__name__, exc))
        state = resolve_and_apply()

    # One bounded repair pass updates whichever durable status artifact
    # survived the first pass. There is deliberately no retry loop.
    if metadata_ok and not manifest_ok:
        try:
            writer(metadata_path, metadata)
        except Exception as exc:
            finalization_errors.append(
                "metadata_repair: %s: %s" % (type(exc).__name__, exc))
    elif manifest_ok and not metadata_ok:
        try:
            writer(manifest_path, manifest_payload(state))
        except Exception as exc:
            finalization_errors.append(
                "manifest_repair: %s: %s" % (type(exc).__name__, exc))

    state = resolve_and_apply()
    state["metadata_write_succeeded"] = metadata_ok
    state["manifest_write_succeeded"] = manifest_ok
    return state


def validate_operational_timing_metadata(metadata, camera_required=False):
    required = list(CORE_OPERATIONAL_TIMING_FIELDS)
    if camera_required:
        required.extend(CAMERA_OPERATIONAL_TIMING_FIELDS)
    missing = [field for field in required if metadata.get(field) is None]
    if missing:
        raise ValueError("Missing operational timing metadata: %s" % ", ".join(missing))
    ordered = [
        "operator_gate_enter_monotonic_ns", "operator_gate_release_monotonic_ns",
        "pre_start_monotonic_ns", "pre_end_monotonic_ns",
        "task_start_monotonic_ns", "task_end_monotonic_ns",
        "post_start_monotonic_ns", "post_end_monotonic_ns",
    ]
    values = [int(metadata[field]) for field in ordered]
    if values != sorted(values):
        raise ValueError("Operational phase timestamps are out of order.")
    return True


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
        "suction_scheduled": trial.get("suction_scheduled", ""),
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
        "suction_scheduled": "",
        "rewarded_cue_presentation_ordinal": "",
    }


def _optional_positive_finite_config_value(config, name):
    value = config.get(name)
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("%s must be null or a finite positive number." % name) from exc
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("%s must be null or a finite positive number." % name)
    return value


def load_reward_volume_config(config):
    per_train = _optional_positive_finite_config_value(
        config, "reward_volume_ul_per_train")
    maximum = _optional_positive_finite_config_value(
        config, "maximum_session_reward_ul")
    if maximum is not None and per_train is None:
        raise RuntimeError(
            "maximum_session_reward_ul requires reward_volume_ul_per_train.")
    config["reward_volume_ul_per_train"] = per_train
    config["maximum_session_reward_ul"] = maximum
    config["reward_volume_cap_enabled"] = bool(
        per_train is not None and maximum is not None)
    return config


class RewardVolumeCapExceeded(RuntimeError):
    pass


class RewardVolumeTracker(object):
    """In-memory reward-volume accounting; no I/O occurs on the fast path."""

    def __init__(self, reward_volume_ul_per_train=None,
                 maximum_session_reward_ul=None,
                 scheduled_reward_train_count=0):
        self.reward_volume_ul_per_train = reward_volume_ul_per_train
        self.maximum_session_reward_ul = maximum_session_reward_ul
        self.reward_volume_cap_enabled = bool(
            reward_volume_ul_per_train is not None
            and maximum_session_reward_ul is not None)
        self.scheduled_reward_train_count = int(scheduled_reward_train_count)
        self.scheduled_reward_dispatched_count = 0
        self.manual_reward_train_count = 0
        self.delivered_reward_train_count = 0
        self.reward_volume_cap_exceeded = False

    @property
    def planned_reward_volume_ul(self):
        if self.reward_volume_ul_per_train is None:
            return None
        return self.scheduled_reward_train_count * self.reward_volume_ul_per_train

    @property
    def estimated_delivered_reward_ul(self):
        if self.reward_volume_ul_per_train is None:
            return None
        return self.delivered_reward_train_count * self.reward_volume_ul_per_train

    def preflight(self):
        if (self.reward_volume_cap_enabled
                and self.planned_reward_volume_ul > self.maximum_session_reward_ul):
            self.reward_volume_cap_exceeded = True
            raise RewardVolumeCapExceeded(
                "Planned reward volume %.3f uL exceeds session cap %.3f uL."
                % (self.planned_reward_volume_ul,
                   self.maximum_session_reward_ul))
        return self

    def check_next_command(self, scheduled=False):
        if not self.reward_volume_cap_enabled:
            return
        future_scheduled = (
            self.scheduled_reward_train_count
            - self.scheduled_reward_dispatched_count
            - (1 if scheduled else 0))
        projected_trains = (
            self.delivered_reward_train_count + 1 + max(0, future_scheduled))
        projected_volume = projected_trains * self.reward_volume_ul_per_train
        if projected_volume > self.maximum_session_reward_ul:
            self.reward_volume_cap_exceeded = True
            kind = "scheduled" if scheduled else "manual"
            raise RewardVolumeCapExceeded(
                "The next %s reward train would raise estimated volume to %.3f uL, "
                "above the %.3f uL session cap."
                % (kind, projected_volume, self.maximum_session_reward_ul))

    def record_dispatched(self, scheduled=False, manual=False):
        if scheduled == manual:
            raise ValueError("Record exactly one reward command kind.")
        self.delivered_reward_train_count += 1
        if scheduled:
            self.scheduled_reward_dispatched_count += 1
        if manual:
            self.manual_reward_train_count += 1

    def summary(self):
        return {
            "reward_volume_cap_enabled": self.reward_volume_cap_enabled,
            "reward_volume_ul_per_train": self.reward_volume_ul_per_train,
            "maximum_session_reward_ul": self.maximum_session_reward_ul,
            "planned_reward_train_count": self.scheduled_reward_train_count,
            "planned_reward_volume_ul": self.planned_reward_volume_ul,
            "manual_reward_train_count": self.manual_reward_train_count,
            "delivered_reward_train_count": self.delivered_reward_train_count,
            "estimated_delivered_reward_ul": self.estimated_delivered_reward_ul,
            "reward_volume_cap_exceeded": self.reward_volume_cap_exceeded,
        }


class TaskWaterTelemetryAccounting(object):
    """Cumulative task-only accounting independent of UDP packet delivery."""

    def __init__(self, reward_volume_ul_per_train=None):
        self.reward_volume_ul_per_train = reward_volume_ul_per_train
        self.verified_trial_indices = set()
        self.contacted_trial_indices = set()

    def record_trial(self, trial_index, reward_delivered, reward_contacted):
        if reward_delivered:
            self.verified_trial_indices.add(int(trial_index))
        if reward_delivered and reward_contacted is True:
            self.contacted_trial_indices.add(int(trial_index))

    def summary(self):
        volume = self.reward_volume_ul_per_train
        return {
            "reward_volume_ul_per_train": volume,
            "task_reward_trains_verified_session": len(self.verified_trial_indices),
            "task_reward_trains_contacted_session": len(self.contacted_trial_indices),
            "task_water_delivered_ul_session": (
                None if volume is None else len(self.verified_trial_indices) * volume
            ),
            "task_water_likely_consumed_ul_session": (
                None if volume is None else len(self.contacted_trial_indices) * volume
            ),
        }


def _telemetry_water_fields(water_accounting):
    if water_accounting is None:
        return {
            "reward_volume_ul_per_train": None,
            "task_reward_trains_verified_session": 0,
            "task_reward_trains_contacted_session": 0,
            "task_water_delivered_ul_session": None,
            "task_water_likely_consumed_ul_session": None,
        }
    return water_accounting.summary()


def build_telemetry_state_payload(phase, trial=None, total_trials=0,
                                  total_blocks=0, eta_sec=None,
                                  water_accounting=None):
    trial = trial or {}
    payload = {
        "phase": str(phase),
        "trial": trial.get("trial_number") if trial else None,
        "total_trials": int(total_trials),
        "block": trial.get("block_number") if trial else None,
        "total_blocks": int(total_blocks),
        "image": trial.get("image_filename") if trial else None,
        "stimulus_role": trial.get("image_role") if trial else None,
        "eta_sec": None if eta_sec is None else max(0.0, float(eta_sec)),
    }
    payload.update(_telemetry_water_fields(water_accounting))
    return payload


def _event_monotonic_ns(event):
    value = event.get("monotonic_ns")
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def recent_trial_gpio_events(all_gpio_events, stim_request_monotonic_ns,
                             stimulus_onset_compensation_sec=0.0):
    """Return only current-trial/recent events needed by telemetry derivation."""
    if stim_request_monotonic_ns in (None, ""):
        return []
    estimated_onset_ns = int(round(
        int(stim_request_monotonic_ns)
        + float(stimulus_onset_compensation_sec) * 1_000_000_000.0
    ))
    lower_ns = estimated_onset_ns - 500_000_000
    recent = []
    for event in reversed(all_gpio_events):
        event_ns = _event_monotonic_ns(event)
        if event_ns is not None and event_ns < lower_ns:
            break
        recent.append(event)
    recent.reverse()
    return recent


def telemetry_lick_times_sec(all_gpio_events, stim_request_monotonic_ns,
                             stimulus_onset_compensation_sec=0.0):
    """Return recent lick onsets in the dashboard's estimated physical window."""
    if stim_request_monotonic_ns in (None, ""):
        return []
    estimated_onset_ns = int(round(
        int(stim_request_monotonic_ns)
        + float(stimulus_onset_compensation_sec) * 1_000_000_000.0
    ))
    lower_ns = estimated_onset_ns - 500_000_000
    upper_ns = estimated_onset_ns + 4_000_000_000
    values = []
    for event in reversed(recent_trial_gpio_events(
            all_gpio_events, stim_request_monotonic_ns,
            stimulus_onset_compensation_sec)):
        event_ns = _event_monotonic_ns(event)
        if event_ns is None:
            continue
        if event_ns < lower_ns:
            break
        if (event.get("event_type") == "lick_onset"
                and event_ns <= upper_ns):
            values.append((event_ns - estimated_onset_ns) / 1e9)
    return sorted(values)


def _first_gpio_event_ns(events, event_type):
    values = [
        _event_monotonic_ns(event) for event in events
        if event.get("event_type") == event_type
    ]
    values = [value for value in values if value is not None]
    return min(values) if values else None


def derive_telemetry_reward_fields(trial, runtime, all_gpio_events,
                                   reward_num_pulses,
                                   recent_events=None):
    recent_events = (
        recent_events if recent_events is not None else
        recent_trial_gpio_events(
            all_gpio_events, runtime.get("stim_request_monotonic_ns"))
    )
    reward_scheduled = bool(trial.get("reward_scheduled"))
    reward_omission = bool(trial.get("reward_omission_scheduled"))
    command_id = runtime.get("reward_command_id", "")
    reward_delivered = False
    reward_contacted = False if not reward_scheduled else None
    command_events = [
        event for event in recent_events
        if command_id and event.get("command_id") == command_id
    ]
    if reward_scheduled and command_id:
        reward_delivered = bool(
            _reward_command_state(
                command_events, command_id, reward_num_pulses
            )["complete"]
        )
        if reward_delivered:
            reward_on_ns = _first_gpio_event_ns(command_events, "reward_valve_on")
            suction_on_ns = _first_gpio_event_ns(
                [
                    event for event in recent_events
                    if event.get("command_id") == runtime.get("suction_command_id", "")
                ],
                "suction_on",
            )
            if reward_on_ns is not None and suction_on_ns is not None:
                reward_contacted = False
                for event in recent_events:
                    if event.get("event_type") != "lick_onset":
                        continue
                    event_ns = _event_monotonic_ns(event)
                    if (event_ns is not None
                            and reward_on_ns <= event_ns < suction_on_ns):
                        reward_contacted = True
                        break
    return {
        "reward_scheduled": reward_scheduled,
        "reward_omission": reward_omission,
        "reward_delivered": reward_delivered,
        "reward_contacted": reward_contacted,
    }


def build_trial_telemetry_payload(trial, runtime, all_gpio_events,
                                  total_trials, total_blocks,
                                  reward_num_pulses,
                                  stimulus_onset_compensation_sec=0.0,
                                  eta_sec=None, water_accounting=None,
                                  reward_fields=None):
    reward_fields = reward_fields or derive_telemetry_reward_fields(
        trial, runtime, all_gpio_events, reward_num_pulses)
    lick_times = telemetry_lick_times_sec(
        all_gpio_events,
        runtime.get("stim_request_monotonic_ns"),
        stimulus_onset_compensation_sec,
    )
    payload = {
        "trial": trial.get("trial_number"),
        "total_trials": int(total_trials),
        "block": trial.get("block_number"),
        "total_blocks": int(total_blocks),
        "image": trial.get("image_filename"),
        "stimulus_role": trial.get("image_role"),
        "reward_scheduled": reward_fields["reward_scheduled"],
        "reward_omission": reward_fields["reward_omission"],
        "reward_delivered": reward_fields["reward_delivered"],
        "reward_contacted": reward_fields["reward_contacted"],
        "anticipatory_lick": any(0.0 <= value < 1.0 for value in lick_times),
        "lick_times_sec": lick_times,
        "lick_time_reference": "estimated_physical_stim_onset",
        "eta_sec": None if eta_sec is None else max(0.0, float(eta_sec)),
    }
    payload.update(_telemetry_water_fields(water_accounting))
    return payload


def publish_completed_trial_telemetry(publisher, trial, runtime,
                                      all_gpio_events, total_trials,
                                      total_blocks, reward_num_pulses,
                                      stimulus_onset_compensation_sec=0.0,
                                      eta_sec=None, water_accounting=None):
    try:
        recent_events = recent_trial_gpio_events(
            all_gpio_events,
            runtime.get("stim_request_monotonic_ns"),
            stimulus_onset_compensation_sec,
        )
        reward_fields = derive_telemetry_reward_fields(
            trial, runtime, all_gpio_events, reward_num_pulses,
            recent_events=recent_events,
        )
        if water_accounting is not None:
            water_accounting.record_trial(
                trial.get("trial_index", -1),
                reward_fields["reward_delivered"],
                reward_fields["reward_contacted"],
            )
        if publisher is None:
            return reward_fields
        publisher.publish_trial(build_trial_telemetry_payload(
            trial, runtime, recent_events, total_trials, total_blocks,
            reward_num_pulses, stimulus_onset_compensation_sec,
            eta_sec, water_accounting, reward_fields=reward_fields))
        return reward_fields
    except Exception:
        _record_telemetry_parent_error(publisher)
        return None


def planned_reward_train_count(trials):
    return sum(1 for trial in trials if trial.get("reward_scheduled"))


def calculate_reward_timing(hardware_config):
    """Calculate nominal reward timing and reject overlap in the GPIO worker."""
    reward_train_duration_sec = (
        int(hardware_config["reward_num_pulses"])
        * float(hardware_config["reward_pulse_on_sec"])
        + max(0, int(hardware_config["reward_num_pulses"]) - 1)
        * float(hardware_config["reward_pulse_off_sec"])
    )
    suction_delay_sec = float(
        hardware_config["suction_delay_from_stim_onset_sec"]
    )
    reward_nominal_start_from_stim_onset_sec = REWARD_DELAY_SEC
    reward_nominal_completion_from_stim_onset_sec = (
        REWARD_DELAY_SEC + reward_train_duration_sec
    )
    reward_to_suction_nominal_gap_sec = (
        suction_delay_sec - reward_nominal_completion_from_stim_onset_sec
    )
    reward_train_extension_into_iti_sec = max(
        0.0,
        reward_train_duration_sec - POST_REWARD_STIM_SEC,
    )
    if reward_nominal_completion_from_stim_onset_sec > suction_delay_sec:
        raise RuntimeError(
            "Configured reward train would nominally finish at %.3f s after "
            "stimulus onset, later than suction onset at %.3f s. Reward and "
            "suction must not overlap in the single GPIO worker."
            % (
                reward_nominal_completion_from_stim_onset_sec,
                suction_delay_sec,
            )
        )
    return {
        "reward_train_duration_sec": reward_train_duration_sec,
        "reward_nominal_start_from_stim_onset_sec": (
            reward_nominal_start_from_stim_onset_sec
        ),
        "reward_nominal_completion_from_stim_onset_sec": (
            reward_nominal_completion_from_stim_onset_sec
        ),
        "reward_to_suction_nominal_gap_sec": reward_to_suction_nominal_gap_sec,
        "reward_train_extension_into_iti_sec": (
            reward_train_extension_into_iti_sec
        ),
        "reward_train_extends_into_iti": (
            reward_train_extension_into_iti_sec > 0.0
        ),
    }


def _calibration_field_present(value):
    return value not in (None, "")


def _validated_refresh_count(value, field_name):
    if isinstance(value, bool):
        raise RuntimeError("%s must be an integer refresh count." % field_name)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise RuntimeError("%s must be an integer refresh count." % field_name)
    if not math.isfinite(numeric) or numeric != int(numeric) or int(numeric) < 1:
        raise RuntimeError("%s must be an integer refresh count of at least 1." % field_name)
    return int(numeric)


def resolve_display_timing_calibration(hardware_config):
    """Resolve optional Box151 photodiode calibration without changing targets."""
    calibration_id = hardware_config.get("display_timing_calibration_id")
    refresh_rate = hardware_config.get("display_timing_calibration_refresh_rate_hz")
    segment1_refreshes = hardware_config.get("stim_segment1_refreshes")
    segment2_refreshes = hardware_config.get("stim_segment2_refreshes")
    compensation = hardware_config.get("stimulus_onset_compensation_sec")
    any_calibration_field = any(
        _calibration_field_present(value)
        for value in (calibration_id, refresh_rate, segment1_refreshes, segment2_refreshes)
    ) or (compensation not in (None, "", 0, 0.0))

    if not any_calibration_field:
        segment1_refreshes = base.refreshes_for_seconds(REWARD_DELAY_SEC)
        segment2_refreshes = base.refreshes_for_seconds(POST_REWARD_STIM_SEC)
        return {
            "calibration_enabled": False,
            "calibration_id": None,
            "calibration_refresh_rate_hz": float(base.REFRESH_RATE_HZ),
            "segment1_refreshes": segment1_refreshes,
            "segment2_refreshes": segment2_refreshes,
            "stim_total_refreshes": segment1_refreshes + segment2_refreshes,
            "segment1_programmed_duration_sec": segment1_refreshes / float(base.REFRESH_RATE_HZ),
            "segment2_programmed_duration_sec": segment2_refreshes / float(base.REFRESH_RATE_HZ),
            "total_programmed_duration_sec": (segment1_refreshes + segment2_refreshes) / float(base.REFRESH_RATE_HZ),
            "stimulus_onset_compensation_sec": 0.0,
        }

    if calibration_id in (None, "") or not str(calibration_id).strip():
        raise RuntimeError("display_timing_calibration_id is required when calibration is configured.")
    try:
        refresh_rate = float(refresh_rate)
    except (TypeError, ValueError):
        raise RuntimeError("display_timing_calibration_refresh_rate_hz must be finite and positive.")
    if not math.isfinite(refresh_rate) or refresh_rate <= 0:
        raise RuntimeError("display_timing_calibration_refresh_rate_hz must be finite and positive.")
    if not math.isclose(refresh_rate, float(base.REFRESH_RATE_HZ), rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            "display timing calibration refresh rate %.9g Hz does not match configured base refresh rate %.9g Hz."
            % (refresh_rate, float(base.REFRESH_RATE_HZ))
        )
    segment1_refreshes = _validated_refresh_count(segment1_refreshes, "stim_segment1_refreshes")
    segment2_refreshes = _validated_refresh_count(segment2_refreshes, "stim_segment2_refreshes")
    try:
        compensation = float(0.0 if compensation in (None, "") else compensation)
    except (TypeError, ValueError):
        raise RuntimeError("stimulus_onset_compensation_sec must be finite and nonnegative.")
    if not math.isfinite(compensation) or compensation < 0:
        raise RuntimeError("stimulus_onset_compensation_sec must be finite and nonnegative.")
    total_refreshes = segment1_refreshes + segment2_refreshes
    return {
        "calibration_enabled": True,
        "calibration_id": str(calibration_id).strip(),
        "calibration_refresh_rate_hz": refresh_rate,
        "segment1_refreshes": segment1_refreshes,
        "segment2_refreshes": segment2_refreshes,
        "stim_total_refreshes": total_refreshes,
        "segment1_programmed_duration_sec": segment1_refreshes / refresh_rate,
        "segment2_programmed_duration_sec": segment2_refreshes / refresh_rate,
        "total_programmed_duration_sec": total_refreshes / refresh_rate,
        "stimulus_onset_compensation_sec": compensation,
    }


def _record_reward_volume_block(event_log_path, context, exc):
    fields = dict(context or {})
    fields["notes"] = str(exc)
    append_event(
        event_log_path,
        exact_timestamp_event(
            "reward_volume_cap_blocked_command",
            **fields,
        ),
    )


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
        "suction_pin_bcm",
        "suction_delay_from_stim_onset_sec",
        "suction_duration_sec",
        "reward_pulse_on_sec",
        "reward_pulse_off_sec",
        "reward_num_pulses",
        "reward_volume_ul_per_train",
        "maximum_session_reward_ul",
    ]
    missing = [name for name in required if name not in config]
    if missing:
        raise RuntimeError("Hardware config is missing: %s" % ", ".join(missing))

    config["simulate_gpio"] = bool(simulate_gpio or config.get("simulate_gpio", False))
    if config["simulate_gpio"] and config.get("reward_pulse_on_sec") in (None, ""):
        config["reward_pulse_on_sec"] = 0.002
    if config["simulate_gpio"] and config.get("suction_duration_sec") in (None, ""):
        config["suction_duration_sec"] = 0.05
    if config.get("reward_pulse_on_sec") in (None, ""):
        raise RuntimeError(
            "reward_pulse_on_sec is unset. Copy the calibrated "
            "session_info['solenoid_blink_duration'] value from the working "
            "Go/NoGo setup; do not guess this valve timing."
        )

    config["reward_pin_bcm"] = int(config["reward_pin_bcm"])
    config["lick_pin_bcm"] = int(config["lick_pin_bcm"])
    config["suction_pin_bcm"] = int(config["suction_pin_bcm"])
    if min(config["reward_pin_bcm"], config["lick_pin_bcm"], config["suction_pin_bcm"]) < 0:
        raise RuntimeError("Reward, lick, and suction GPIO pins must be nonnegative integers.")
    config["reward_pulse_on_sec"] = float(config["reward_pulse_on_sec"])
    config["reward_pulse_off_sec"] = float(config["reward_pulse_off_sec"])
    config["reward_num_pulses"] = int(config["reward_num_pulses"])
    if not math.isfinite(config["reward_pulse_on_sec"]):
        raise RuntimeError("reward_pulse_on_sec must be finite.")
    if not math.isfinite(config["reward_pulse_off_sec"]):
        raise RuntimeError("reward_pulse_off_sec must be finite.")
    if len({config["reward_pin_bcm"], config["lick_pin_bcm"], config["suction_pin_bcm"]}) != 3:
        raise RuntimeError("Reward, lick, and suction GPIO pins must be distinct.")
    if config["reward_pulse_on_sec"] <= 0:
        raise RuntimeError("reward_pulse_on_sec must be positive.")
    if config["reward_pulse_off_sec"] < 0:
        raise RuntimeError("reward_pulse_off_sec cannot be negative.")
    if config["reward_num_pulses"] < 1:
        raise RuntimeError("reward_num_pulses must be at least 1.")
    config["suction_delay_from_stim_onset_sec"] = float(config["suction_delay_from_stim_onset_sec"])
    if not math.isfinite(config["suction_delay_from_stim_onset_sec"]) or config["suction_delay_from_stim_onset_sec"] < STIM_DURATION_SEC:
        raise RuntimeError("suction delay must be finite and at least stimulus duration.")
    if config.get("suction_duration_sec") in (None, ""):
        raise RuntimeError("suction_duration_sec must be set for real GPIO.")
    config["suction_duration_sec"] = float(config["suction_duration_sec"])
    if not math.isfinite(config["suction_duration_sec"]) or config["suction_duration_sec"] <= 0:
        raise RuntimeError("suction_duration_sec must be finite and positive.")
    earliest_next = STIM_DURATION_SEC + DEFAULT_ITI_MIN_SEC
    if (config["suction_delay_from_stim_onset_sec"] + config["suction_duration_sec"]
            > earliest_next - MINIMUM_CLEAN_POST_SUCTION_SEC):
        raise RuntimeError("Suction does not leave the required 0.75 s clean gray interval.")
    return load_reward_volume_config(config)


def get_git_commit(
):
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


def build_stimulus_segment_raws(
    rpg_module, raw_cache_root, assignment_rows, display_timing=None
):
    display_timing = display_timing or resolve_display_timing_calibration({})
    segment1_refreshes = display_timing["segment1_refreshes"]
    segment2_refreshes = display_timing["segment2_refreshes"]
    first_dir = base.ensure_dir(raw_cache_root / ("stim_reward_pre_%dr" % segment1_refreshes))
    second_dir = base.ensure_dir(raw_cache_root / ("stim_reward_post_%dr" % segment2_refreshes))
    raw_paths = {}
    for row in assignment_rows:
        image_path = Path(row["image_path"])
        canvas = base.build_canvas(
            image_path, base.SCREEN_RESOLUTION, photodiode_on=True
        )
        first_path = first_dir / (image_path.stem + "_reward_pre_%dr.raw" % segment1_refreshes)
        second_path = second_dir / (image_path.stem + "_reward_post_%dr.raw" % segment2_refreshes)
        base.convert_canvas_to_rpg_raw_refreshes(
            rpg_module, canvas, first_path, segment1_refreshes
        )
        base.convert_canvas_to_rpg_raw_refreshes(
            rpg_module, canvas, second_path, segment2_refreshes
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


def wait_until(deadline_monotonic, gpio_client, event_log_path, all_gpio_events, status_callback=None):
    while True:
        remaining = deadline_monotonic - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(BACKGROUND_POLL_SEC, remaining))
        log_drained_gpio_events(gpio_client, event_log_path, all_gpio_events)
        if status_callback: status_callback(max(0.0, deadline_monotonic - time.monotonic()))


def hold_background(
    phase,
    duration_sec,
    gpio_client,
    event_log_path,
    all_gpio_events,
    pending_reward_checks=None,
    status_callback=None,
    on_pending_checks_complete=None,
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
    requested_deadline = start_monotonic + float(duration_sec)
    required_hardware_deadline = requested_deadline
    for check in pending_reward_checks or []:
        if "suction_target" in check:
            wait_until(check["suction_target"], gpio_client, event_log_path, all_gpio_events)
            command_id = gpio_client.trigger_suction(check["context"])
            if check.get("runtime") is not None:
                check["runtime"]["suction_command_id"] = command_id
            suction_deadline = time.monotonic() + float(check.get("duration_sec", 0.0))
            required_hardware_deadline = max(required_hardware_deadline, suction_deadline)
            verify_suction_command(gpio_client, command_id, event_log_path, all_gpio_events,
                                   max(0.01, suction_deadline - time.monotonic() + 1.0))
            continue
        # Required reward verification is independent of the requested POST
        # duration. A zero/expired POST must not skip this operation.
        verify_reward_command(
            gpio_client,
            check["command_id"],
            int(check["expected_num_pulses"]),
            event_log_path,
            all_gpio_events,
            timeout_sec=float(check.get("timeout_sec", 5.0)),
        )
    if on_pending_checks_complete:
        on_pending_checks_complete()
    wait_until(
        max(requested_deadline, required_hardware_deadline),
        gpio_client,
        event_log_path,
        all_gpio_events,
        status_callback=status_callback,
    )
    actual = time.monotonic() - start_monotonic
    end = exact_timestamp_event(
        phase + "_end",
        phase=phase,
        planned_duration_sec=float(duration_sec),
        notes=("requested_duration_sec=%.9f; actual_duration_sec=%.9f"
               % (float(duration_sec), actual)),
    )
    append_event(event_log_path, end)
    return actual


def _reward_command_state(all_gpio_events, command_id, expected_num_pulses):
    relevant = [
        event
        for event in all_gpio_events
        if event.get("command_id") == command_id
        and event.get("event_type") in {
            "reward_command_received",
            "reward_valve_on",
            "reward_valve_off",
            "reward_complete",
        }
    ]
    received = [event for event in relevant if event.get("event_type") == "reward_command_received"]
    valve_on = [event for event in relevant if event.get("event_type") == "reward_valve_on"]
    valve_off = [event for event in relevant if event.get("event_type") == "reward_valve_off"]
    complete = [event for event in relevant if event.get("event_type") == "reward_complete"]
    return {
        "command_id": command_id,
        "expected_num_pulses": int(expected_num_pulses),
        "reward_command_received_count": len(received),
        "reward_valve_on_count": len(valve_on),
        "reward_valve_off_count": len(valve_off),
        "reward_complete_count": len(complete),
        "complete": (
            len(received) == 1
            and len(complete) == 1
            and len(valve_on) == int(expected_num_pulses)
            and len(valve_off) == int(expected_num_pulses)
        ),
    }


def verify_reward_command(
    gpio_client,
    command_id,
    expected_num_pulses,
    event_log_path,
    all_gpio_events,
    timeout_sec,
):
    deadline = time.monotonic() + float(timeout_sec)
    while True:
        log_drained_gpio_events(gpio_client, event_log_path, all_gpio_events)
        state = _reward_command_state(all_gpio_events, command_id, expected_num_pulses)
        if state["reward_command_received_count"] > 1:
            raise RuntimeError("Reward command %s was received more than once." % command_id)
        if state["reward_complete_count"] > 1:
            raise RuntimeError("Reward command %s completed more than once." % command_id)
        if state["reward_valve_on_count"] > int(expected_num_pulses):
            raise RuntimeError("Reward command %s produced too many valve-on events." % command_id)
        if state["reward_valve_off_count"] > int(expected_num_pulses):
            raise RuntimeError("Reward command %s produced too many valve-off events." % command_id)
        if state["complete"]:
            state.update({"reward_command_verified": True})
            return state
        if not gpio_client.is_alive():
            raise RuntimeError("GPIO worker died before reward command %s was verified." % command_id)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(BACKGROUND_POLL_SEC, remaining))

    log_drained_gpio_events(gpio_client, event_log_path, all_gpio_events)
    state = _reward_command_state(all_gpio_events, command_id, expected_num_pulses)
    raise RuntimeError(
        "Reward command %s did not reach the expected event pattern before the deadline: %s"
        % (command_id, json.dumps(state, sort_keys=True))
    )


def verify_suction_command(gpio_client, command_id, event_log_path, all_gpio_events, timeout_sec):
    deadline = time.monotonic() + float(timeout_sec)
    while time.monotonic() < deadline:
        log_drained_gpio_events(gpio_client, event_log_path, all_gpio_events)
        matching = [e for e in all_gpio_events if e.get("command_id") == command_id]
        counts = {name: sum(1 for e in matching if e.get("event_type") == name)
                  for name in ("suction_command_received", "suction_on", "suction_off", "suction_complete")}
        if any(value > 1 for value in counts.values()):
            raise RuntimeError("Suction command %s produced duplicate events: %s" % (command_id, counts))
        if all(value == 1 for value in counts.values()):
            return counts
        if not gpio_client.is_alive():
            raise RuntimeError("GPIO worker died before suction command %s was verified." % command_id)
        time.sleep(min(BACKGROUND_POLL_SEC, deadline - time.monotonic()))
    log_drained_gpio_events(gpio_client, event_log_path, all_gpio_events)
    raise RuntimeError("Suction command %s did not complete before the ITI deadline." % command_id)


def suction_target_from_stim_request(
    stim_request_monotonic_ns, suction_delay_sec, stimulus_onset_compensation_sec=0.0
):
    return (
        float(stim_request_monotonic_ns) / 1e9
        + float(stimulus_onset_compensation_sec)
        + float(suction_delay_sec)
    )


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
    reward_num_pulses,
    reward_verification_timeout_sec,
    suction_delay_sec,
    suction_duration_sec,
    status_callback=None,
    reward_volume_tracker=None,
    stimulus_onset_compensation_sec=0.0,
    telemetry_publisher=None,
    telemetry_state_reporter=None,
    water_accounting=None,
    total_blocks=0,
    post_background_sec=0.0,
):
    runtime_by_trial = {}
    pending_final_reward_checks = []
    task_start_monotonic = time.monotonic()
    total_trials = len(trials)

    for completed_count, trial in enumerate(trials, start=1):
        trial_index = trial["trial_index"]
        runtime = {
            "trial_executed": False,
            "stim_presented": False,
            "trial_completed": False,
            "reward_command_id": "",
            "suction_command_id": "",
            "stim_request_unix_ns": "",
            "stim_request_monotonic_ns": "",
            "stim_segment1_return_unix_ns": "",
            "stim_segment2_request_unix_ns": "",
            "stim_offset_request_unix_ns": "",
            "stim_offset_monotonic_ns": "",
            "reward_boundary_monotonic_ns": "",
            "segment_boundary_gap_sec": "",
        }
        runtime_by_trial[trial_index] = runtime

        context = trial_context(trial, "stimulus")
        gpio_client.set_context(context)
        log_drained_gpio_events(gpio_client, event_log_path, all_gpio_events)

        image_raws = loaded_raws[trial["image_filename"]]
        image_paths = raw_paths[trial["image_filename"]]

        safe_report_telemetry_state(
            telemetry_state_reporter,
            "STIMULUS",
            trial,
            total_trials,
            total_blocks,
            planned_task_remaining_seconds(trials, completed_count - 1)
            + float(post_background_sec),
            water_accounting,
            force=True,
        )
        runtime["trial_executed"] = True
        runtime["stim_request_monotonic_ns"] = time.monotonic_ns()
        stim_request_timestamp = base.capture_timestamp()
        runtime["stim_request_unix_ns"] = stim_request_timestamp["unix_ns"]

        first_perf, first_timing = base.display_raw_with_timing(
            screen, image_raws["first"]
        )

        boundary_timestamp = base.capture_timestamp()
        boundary_monotonic_ns = time.monotonic_ns()
        runtime["reward_boundary_monotonic_ns"] = boundary_monotonic_ns
        reward_command_id = ""
        if trial["reward_scheduled"]:
            try:
                if reward_volume_tracker is not None:
                    reward_volume_tracker.check_next_command(scheduled=True)
                reward_command_id = gpio_client.trigger_reward(context)
                if reward_volume_tracker is not None:
                    reward_volume_tracker.record_dispatched(scheduled=True)
            except RewardVolumeCapExceeded as exc:
                # Exceptional safety path only: remove the stimulus before any
                # file I/O or exception propagation. Normal reward timing does
                # not enter this branch.
                blocked_offset_monotonic_ns = time.monotonic_ns()
                blocked_background_perf, blocked_background_timing = (
                    base.display_raw_with_timing(screen, loaded_background_raw)
                )
                runtime["stim_presented"] = True
                runtime["stim_segment1_return_unix_ns"] = first_timing["return_unix_ns"]
                runtime["stim_offset_request_unix_ns"] = blocked_background_timing["request_unix_ns"]
                runtime["stim_offset_monotonic_ns"] = blocked_offset_monotonic_ns

                first_row = make_display_event(
                    "stimulus_segment_1", trial, "first_1p0_sec",
                    image_paths["first"], REWARD_DELAY_SEC,
                    first_perf, first_timing)
                first_row["notes"] = "partial_stimulus; reward_volume_cap_abort"
                append_event(event_log_path, first_row)

                blocked_offset_row = trial_context(trial, "stimulus")
                blocked_offset_row.update({
                    "utc_iso": blocked_background_timing["request_utc_iso"],
                    "unix_time_utc_sec": blocked_background_timing["request_unix_sec"],
                    "unix_time_ns": blocked_background_timing["request_unix_ns"],
                    "monotonic_ns": blocked_background_timing["request_perf_counter_ns"],
                    "event_type": "stimulus_offset_reward_volume_cap_abort",
                    "segment_name": "gray_photodiode_off",
                    "raw_path": str(background_raw_path),
                    "planned_duration_sec": 0.0,
                    "display_request_unix_ns": blocked_background_timing["request_unix_ns"],
                    "display_return_unix_ns": blocked_background_timing["return_unix_ns"],
                    "display_request_perf_counter_ns": blocked_background_timing["request_perf_counter_ns"],
                    "display_return_perf_counter_ns": blocked_background_timing["return_perf_counter_ns"],
                    "display_call_duration_sec": "%.9f" % blocked_background_timing["duration_sec"],
                    "start_time_unix": getattr(blocked_background_perf, "start_time", ""),
                    "mean_interframe_us": getattr(blocked_background_perf, "mean_interframe", ""),
                    "stddev_interframe_us": getattr(blocked_background_perf, "stddev_interframe", ""),
                    "notes": "segment_2_not_displayed; reward_command_not_sent",
                })
                append_event(event_log_path, blocked_offset_row)
                _record_reward_volume_block(event_log_path, context, exc)
                raise
        runtime["reward_command_id"] = reward_command_id
        runtime["suction_scheduled"] = bool(trial.get("suction_scheduled", False))

        second_perf, second_timing = base.display_raw_with_timing(
            screen, image_raws["second"]
        )
        segment_boundary_gap_sec = (
            second_timing["request_perf_counter_ns"]
            - first_timing["return_perf_counter_ns"]
        ) / 1_000_000_000.0

        background_transition_monotonic_ns = time.monotonic_ns()
        background_perf, background_timing = base.display_raw_with_timing(
            screen, loaded_background_raw
        )
        runtime["stim_presented"] = True
        runtime["stim_offset_request_unix_ns"] = background_timing["request_unix_ns"]
        runtime["stim_offset_monotonic_ns"] = background_transition_monotonic_ns
        runtime["segment_boundary_gap_sec"] = segment_boundary_gap_sec

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

        if trial["reward_scheduled"]:
            reward_check = {
                "command_id": reward_command_id,
                "expected_num_pulses": reward_num_pulses,
                "timeout_sec": reward_verification_timeout_sec,
            }
        else:
            reward_check = None
        suction_check = None
        if trial.get("suction_scheduled"):
            suction_check = {
                "target": suction_target_from_stim_request(
                    runtime["stim_request_monotonic_ns"],
                    suction_delay_sec,
                    stimulus_onset_compensation_sec,
                ),
                "duration_sec": float(suction_duration_sec),
            }

        # Populate final-trial segment timing before the early POST return.
        runtime["stim_segment1_return_unix_ns"] = first_timing["return_unix_ns"]
        runtime["stim_segment2_request_unix_ns"] = second_timing["request_unix_ns"]

        if completed_count == total_trials:
            runtime["trial_completed"] = True
            runtime["stim_offset_request_unix_ns"] = background_timing["request_unix_ns"]
            runtime["stim_offset_monotonic_ns"] = background_transition_monotonic_ns
            final_offset_row = trial_context(trial, "stimulus")
            final_offset_row.update(
                {
                    "utc_iso": background_timing["request_utc_iso"],
                    "unix_time_utc_sec": background_timing["request_unix_sec"],
                    "unix_time_ns": background_timing["request_unix_ns"],
                    "monotonic_ns": background_timing["request_perf_counter_ns"],
                    "event_type": "stimulus_offset",
                    "segment_name": "gray_photodiode_off",
                    "raw_path": str(background_raw_path),
                    "planned_duration_sec": 0.0,
                    "planned_iti_duration_sec": trial["planned_iti_duration_sec"],
                    "display_request_unix_ns": background_timing["request_unix_ns"],
                    "display_return_unix_ns": background_timing["return_unix_ns"],
                    "display_request_perf_counter_ns": background_timing["request_perf_counter_ns"],
                    "display_return_perf_counter_ns": background_timing["return_perf_counter_ns"],
                    "display_call_duration_sec": "%.9f" % background_timing["duration_sec"],
                    "start_time_unix": getattr(background_perf, "start_time", ""),
                    "mean_interframe_us": getattr(background_perf, "mean_interframe", ""),
                    "stddev_interframe_us": getattr(background_perf, "stddev_interframe", ""),
                    "notes": "final_planned_iti_skipped; POST begins immediately",
                }
            )
            append_event(event_log_path, final_offset_row)
            if reward_check is not None:
                pending_final_reward_checks.append(reward_check)
            if suction_check is not None:
                pending_final_reward_checks.append({"suction_target": suction_check["target"], "duration_sec": suction_check["duration_sec"], "context": context, "runtime": runtime})
            sys.stdout.write("\n")
            sys.stdout.flush()
            return runtime_by_trial, pending_final_reward_checks

        gpio_client.set_context(trial_context(trial, "iti"))
        log_drained_gpio_events(gpio_client, event_log_path, all_gpio_events)
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
        safe_report_telemetry_state(
            telemetry_state_reporter,
            "ITI",
            trial,
            total_trials,
            total_blocks,
            float(trial["planned_iti_duration_sec"])
            + planned_task_remaining_seconds(trials, completed_count)
            + float(post_background_sec),
            water_accounting,
            force=True,
        )

        # The ITI deadline is anchored to gray onset. Verification and suction
        # are serviced inside this interval and may not extend it.
        iti_start_monotonic = background_transition_monotonic_ns / 1e9
        iti_deadline = iti_start_monotonic + float(trial["planned_iti_duration_sec"])

        def iti_status_callback(_wait_remaining):
            if status_callback:
                status_callback(
                    completed_count,
                    total_trials,
                    planned_task_remaining_during_iti(
                        trials, completed_count - 1, iti_deadline),
                )
            safe_report_telemetry_state(
                telemetry_state_reporter,
                "ITI",
                trial,
                total_trials,
                total_blocks,
                planned_task_remaining_during_iti(
                    trials, completed_count - 1, iti_deadline)
                + float(post_background_sec),
                water_accounting,
            )

        if reward_check is not None:
            verify_reward_command(
                gpio_client,
                reward_check["command_id"],
                reward_check["expected_num_pulses"],
                event_log_path,
                all_gpio_events,
                timeout_sec=min(reward_check["timeout_sec"], max(0.01, iti_deadline - time.monotonic())),
            )
        if suction_check is not None:
            wait_until(
                suction_check["target"], gpio_client, event_log_path,
                all_gpio_events,
                status_callback=(
                    iti_status_callback
                    if status_callback is not None or telemetry_state_reporter is not None
                    else None
                ),
            )
            suction_command_id = gpio_client.trigger_suction(context)
            runtime["suction_command_id"] = suction_command_id
            verify_suction_command(gpio_client, suction_command_id, event_log_path, all_gpio_events,
                                   max(0.01, iti_deadline - time.monotonic()))
        log_drained_gpio_events(gpio_client, event_log_path, all_gpio_events)
        wait_until(
            iti_deadline,
            gpio_client,
            event_log_path,
            all_gpio_events,
            status_callback=(
                iti_status_callback
                if status_callback is not None or telemetry_state_reporter is not None
                else None
            ),
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
        runtime["trial_completed"] = True

        publish_completed_trial_telemetry(
            telemetry_publisher,
            trial,
            runtime,
            all_gpio_events,
            total_trials,
            total_blocks,
            reward_num_pulses,
            stimulus_onset_compensation_sec,
            planned_task_remaining_seconds(trials, completed_count)
            + float(post_background_sec),
            water_accounting,
        )

        if not status_callback:
            remaining = planned_task_remaining_seconds(trials, completed_count)
            sys.stdout.write("\rTASK %d/%d (%.1f%%), planned task remaining %s   " %
                             (completed_count, total_trials, 100.0 * completed_count / float(total_trials), base.format_seconds(remaining)))
            sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()
    return runtime_by_trial, pending_final_reward_checks



def _count_events_in_window(event_ns, onset_ns, start_sec, stop_sec):
    start_ns = onset_ns + int(round(start_sec * 1_000_000_000))
    stop_ns = onset_ns + int(round(stop_sec * 1_000_000_000))
    return sum(1 for value in event_ns if start_ns <= value < stop_ns)


def _blank_trial_summary(trial):
    return {
        "trial_index": trial["trial_index"],
        "trial_number": trial["trial_number"],
        "block_number": trial["block_number"],
        "image_role": trial["image_role"],
        "image_category": trial["image_category"],
        "image_id": trial["image_id"],
        "image_filename": trial["image_filename"],
        "reward_eligible": trial["reward_eligible"],
        "reward_scheduled": trial["reward_scheduled"],
        "reward_omission_scheduled": trial["reward_omission_scheduled"],
        "suction_scheduled": trial.get("suction_scheduled", False),
        "trial_executed": False,
        "stim_presented": False,
        "trial_completed": False,
        "planned_iti_duration_sec": trial["planned_iti_duration_sec"],
        "stim_request_unix_ns": "",
        "stim_request_monotonic_ns": "",
        "stim_segment1_return_unix_ns": "",
        "stim_segment2_request_unix_ns": "",
        "stim_offset_request_unix_ns": "",
        "stim_offset_monotonic_ns": "",
        "segment_boundary_gap_sec": "",
        "reward_command_id": "",
        "suction_command_id": "",
        "suction_command_unix_ns": "",
        "suction_command_monotonic_ns": "",
        "suction_on_unix_ns": "",
        "suction_on_monotonic_ns": "",
        "suction_off_unix_ns": "",
        "suction_off_monotonic_ns": "",
        "suction_complete_unix_ns": "",
        "suction_complete_monotonic_ns": "",
        "software_suction_delay_sec": "",
        "estimated_suction_delay_from_physical_onset_sec": "",
        "first_reward_valve_on_unix_ns": "",
        "actual_reward_delay_from_software_stim_request_sec": "",
        "lick_count_pre_0p5_sec": "",
        "lick_count_0_to_0p5_sec": "",
        "lick_count_0_to_1p0_sec": "",
        "lick_count_0p5_to_1p0_sec": "",
        "lick_count_1p0_to_1p5_sec": "",
        "lick_count_1p5_to_2p0_sec": "",
        "lick_count_2p0_to_3p0_sec": "",
        "lick_count_3p0_to_3p5_sec": "",
        "lick_count_3p5_to_4p0_sec": "",
        "notes": "",
    }


def build_trial_summary(
    trials, runtime_by_trial, all_gpio_events, stimulus_onset_compensation_sec=0.0
):
    all_lick_monotonic_ns = sorted(
        int(event["monotonic_ns"])
        for event in all_gpio_events
        if event.get("event_type") == "lick_onset"
        and event.get("monotonic_ns") not in (None, "")
    )
    reward_events_by_command_id = {}
    suction_events_by_command_id = {}
    for event in all_gpio_events:
        command_id = event.get("command_id", "")
        if not command_id:
            continue
        if event.get("event_type") in (
            "reward_command_received",
            "reward_valve_on",
            "reward_valve_off",
            "reward_complete",
        ):
            reward_events_by_command_id.setdefault(command_id, []).append(event)
        if event.get("event_type") in ("suction_command_received", "suction_on", "suction_off", "suction_complete"):
            suction_events_by_command_id.setdefault(command_id, []).append(event)

    rows = []
    for trial in trials:
        trial_index = trial["trial_index"]
        runtime = runtime_by_trial.get(trial_index)
        row = _blank_trial_summary(trial)
        if not runtime:
            rows.append(row)
            continue

        row["trial_executed"] = bool(runtime.get("trial_executed", False))
        row["stim_presented"] = bool(runtime.get("stim_presented", False))
        row["trial_completed"] = bool(runtime.get("trial_completed", False))
        row["stim_request_unix_ns"] = runtime.get("stim_request_unix_ns", "")
        row["stim_request_monotonic_ns"] = runtime.get("stim_request_monotonic_ns", "")
        row["stim_segment1_return_unix_ns"] = runtime.get("stim_segment1_return_unix_ns", "")
        row["stim_segment2_request_unix_ns"] = runtime.get("stim_segment2_request_unix_ns", "")
        row["stim_offset_request_unix_ns"] = runtime.get("stim_offset_request_unix_ns", "")
        row["stim_offset_monotonic_ns"] = runtime.get("stim_offset_monotonic_ns", "")
        row["segment_boundary_gap_sec"] = runtime.get("segment_boundary_gap_sec", "")
        row["reward_command_id"] = runtime.get("reward_command_id", "")
        row["suction_command_id"] = runtime.get("suction_command_id", "")
        suction_events = suction_events_by_command_id.get(row["suction_command_id"], [])
        for event_type, prefix in (("suction_command_received", "suction_command"), ("suction_on", "suction_on"), ("suction_off", "suction_off"), ("suction_complete", "suction_complete")):
            match = next((e for e in suction_events if e.get("event_type") == event_type), None)
            if match:
                row[prefix + "_unix_ns"] = match.get("unix_time_ns", "")
                row[prefix + "_monotonic_ns"] = match.get("monotonic_ns", "")
        if row["suction_on_monotonic_ns"] not in ("", None) and row["stim_request_monotonic_ns"] not in ("", None):
            row["software_suction_delay_sec"] = (int(row["suction_on_monotonic_ns"]) - int(row["stim_request_monotonic_ns"])) / 1e9
            row["estimated_suction_delay_from_physical_onset_sec"] = (
                row["software_suction_delay_sec"] - float(stimulus_onset_compensation_sec)
            )

        reward_command_id = row["reward_command_id"]
        reward_events = reward_events_by_command_id.get(reward_command_id, []) if reward_command_id else []
        reward_valve_on_events = [event for event in reward_events if event.get("event_type") == "reward_valve_on"]
        if reward_valve_on_events:
            first_reward_event = reward_valve_on_events[0]
            row["first_reward_valve_on_unix_ns"] = first_reward_event.get("unix_time_ns", "")
            stim_on_ns = runtime.get("stim_request_monotonic_ns")
            reward_on_ns = first_reward_event.get("monotonic_ns")
            if stim_on_ns not in (None, "") and reward_on_ns not in (None, ""):
                row["actual_reward_delay_from_software_stim_request_sec"] = (
                    (int(reward_on_ns) - int(stim_on_ns)) / 1_000_000_000.0
                )

        onset_ns = runtime.get("stim_request_monotonic_ns")
        if onset_ns in (None, ""):
            counts = {name: "" for name in ("pre", "early", "full", "late", "post", "after", "consumption", "pre_suction", "post_suction")}
        else:
            onset_ns = int(onset_ns)
            counts = {
                "pre": _count_events_in_window(all_lick_monotonic_ns, onset_ns, -0.5, 0.0),
                "early": _count_events_in_window(all_lick_monotonic_ns, onset_ns, 0.0, 0.5),
                "full": _count_events_in_window(all_lick_monotonic_ns, onset_ns, 0.0, 1.0),
                "late": _count_events_in_window(all_lick_monotonic_ns, onset_ns, 0.5, 1.0),
                "post": _count_events_in_window(all_lick_monotonic_ns, onset_ns, 1.0, 1.5),
                "after": _count_events_in_window(all_lick_monotonic_ns, onset_ns, 1.5, 2.0),
                "consumption": _count_events_in_window(all_lick_monotonic_ns, onset_ns, 2.0, 3.0),
                "pre_suction": _count_events_in_window(all_lick_monotonic_ns, onset_ns, 3.0, 3.5),
                "post_suction": _count_events_in_window(all_lick_monotonic_ns, onset_ns, 3.5, 4.0),
            }
        row["lick_count_pre_0p5_sec"] = counts["pre"]
        row["lick_count_0_to_0p5_sec"] = counts["early"]
        row["lick_count_0_to_1p0_sec"] = counts["full"]
        row["lick_count_0p5_to_1p0_sec"] = counts["late"]
        row["lick_count_1p0_to_1p5_sec"] = counts["post"]
        row["lick_count_1p5_to_2p0_sec"] = counts["after"]
        row["lick_count_2p0_to_3p0_sec"] = counts["consumption"]
        row["lick_count_3p0_to_3p5_sec"] = counts["pre_suction"]
        row["lick_count_3p5_to_4p0_sec"] = counts["post_suction"]
        row["notes"] = (
            "Software-aligned summary; use photodiode/DAQ timing for final neural analysis."
        )
        rows.append(row)
    return rows


def write_rows(path, rows, fields):
    base.write_csv(path, rows, fields)


def estimate_task_seconds(trials):
    if not trials:
        return 0.0
    return (
        len(trials) * STIM_DURATION_SEC
        + sum(float(trial["planned_iti_duration_sec"]) for trial in trials[:-1])
    )


def _series_stat(values, name, statistic_name):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if statistic_name == "median":
        return float(statistics.median(ordered))
    if statistic_name == "p95":
        index = max(0, min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1))
        return float(ordered[index])
    if statistic_name == "max":
        return float(ordered[-1])
    raise ValueError("Unknown statistic: %s" % statistic_name)


def build_session_qc(
    session_id,
    trials,
    trial_summary_rows,
    all_gpio_events,
    reward_num_pulses,
    reward_volume_summary=None,
    camera_data_secured=None,
):
    planned_trial_count = len(trials)
    executed_trial_count = sum(1 for row in trial_summary_rows if row.get("trial_executed"))
    completed_trial_count = sum(1 for row in trial_summary_rows if row.get("trial_completed"))
    planned_reward_count = sum(1 for trial in trials if trial["reward_scheduled"])
    planned_suction_count = sum(1 for trial in trials if trial.get("suction_scheduled"))

    scheduled_reward_command_ids = {
        row.get("reward_command_id")
        for row in trial_summary_rows
        if row.get("reward_command_id")
    }
    reward_events = [
        event for event in all_gpio_events
        if event.get("command_id") in scheduled_reward_command_ids
    ]
    reward_command_received_events = [
        event for event in reward_events if event.get("event_type") == "reward_command_received"
    ]
    reward_complete_events = [
        event for event in reward_events if event.get("event_type") == "reward_complete"
    ]
    reward_valve_on_events = [
        event for event in reward_events if event.get("event_type") == "reward_valve_on"
    ]
    reward_valve_off_events = [
        event for event in reward_events if event.get("event_type") == "reward_valve_off"
    ]
    lick_onset_events = [
        event for event in all_gpio_events if event.get("event_type") == "lick_onset"
    ]
    scheduled_suction_ids = {
        row.get("suction_command_id")
        for row in trial_summary_rows
        if row.get("suction_command_id")
    }
    suction_events = [
        event for event in all_gpio_events
        if event.get("command_id") in scheduled_suction_ids
    ]
    suction_command_received_events = [e for e in suction_events if e.get("event_type") == "suction_command_received"]
    suction_complete_events = [e for e in suction_events if e.get("event_type") == "suction_complete"]
    suction_on_events = [e for e in suction_events if e.get("event_type") == "suction_on"]
    suction_off_events = [e for e in suction_events if e.get("event_type") == "suction_off"]
    all_experimental_suction_received_ids = {
        event.get("command_id")
        for event in all_gpio_events
        if event.get("event_type") == "suction_command_received"
        and event.get("phase") != "manual_suction"
        and event.get("command_id")
    }
    missing_suction_command_ids = sorted("trial_%s" % row.get("trial_index") for row in trial_summary_rows if row.get("suction_scheduled") and not row.get("suction_command_id"))
    unexpected_suction_command_ids = sorted(
        all_experimental_suction_received_ids - scheduled_suction_ids
    )
    scheduled_suction_received_ids = {
        event.get("command_id")
        for event in suction_command_received_events
        if event.get("command_id")
    }
    duplicate_suction_command_ids = sorted(
        command_id
        for command_id in scheduled_suction_received_ids
        if sum(1 for e in suction_command_received_events if e.get("command_id") == command_id) > 1
    )
    incomplete_suction_command_ids = sorted(command_id for command_id in scheduled_suction_ids if command_id and sum(1 for e in suction_complete_events if e.get("command_id") == command_id) != 1)

    scheduled_reward_rows = [
        row for row in trial_summary_rows if row.get("reward_scheduled")
    ]
    scheduled_reward_ids = [
        row.get("reward_command_id", "")
        for row in scheduled_reward_rows
        if row.get("reward_command_id")
    ]
    scheduled_reward_id_set = set(scheduled_reward_ids)
    all_experimental_reward_received_ids = {
        event.get("command_id")
        for event in all_gpio_events
        if event.get("event_type") == "reward_command_received"
        and event.get("command_id")
    }

    missing_reward_command_ids = []
    incomplete_reward_command_ids = []
    duplicate_reward_command_ids = []
    for row in scheduled_reward_rows:
        trial_label = "trial_%s" % row.get("trial_index", "unknown")
        command_id = row.get("reward_command_id", "")
        if not command_id:
            missing_reward_command_ids.append(trial_label)
            continue
        matching = [event for event in all_gpio_events if event.get("command_id") == command_id and event.get("event_type") in {"reward_command_received", "reward_valve_on", "reward_valve_off", "reward_complete"}]
        received = sum(1 for event in matching if event.get("event_type") == "reward_command_received")
        valve_on = sum(1 for event in matching if event.get("event_type") == "reward_valve_on")
        valve_off = sum(1 for event in matching if event.get("event_type") == "reward_valve_off")
        complete = sum(1 for event in matching if event.get("event_type") == "reward_complete")
        if received == 0:
            missing_reward_command_ids.append(command_id)
            continue
        if received > 1:
            duplicate_reward_command_ids.append(command_id)
        if (
            received != 1
            or complete != 1
            or valve_on != int(reward_num_pulses)
            or valve_off != int(reward_num_pulses)
        ):
            incomplete_reward_command_ids.append(command_id)

    unexpected_reward_command_ids = sorted(
        all_experimental_reward_received_ids - scheduled_reward_id_set
    )

    segment_boundary_gap_values = [
        float(row["segment_boundary_gap_sec"])
        for row in trial_summary_rows
        if row.get("segment_boundary_gap_sec") not in ("", None)
    ]
    software_reward_delay_values = [
        float(row["actual_reward_delay_from_software_stim_request_sec"])
        for row in trial_summary_rows
        if row.get("actual_reward_delay_from_software_stim_request_sec") not in ("", None)
    ]

    qc_fail_reasons = []
    if executed_trial_count != planned_trial_count:
        qc_fail_reasons.append(
            "executed_trial_count=%d planned_trial_count=%d" % (executed_trial_count, planned_trial_count)
        )
    if completed_trial_count != planned_trial_count:
        qc_fail_reasons.append(
            "completed_trial_count=%d planned_trial_count=%d" % (completed_trial_count, planned_trial_count)
        )
    if len(reward_command_received_events) != planned_reward_count:
        qc_fail_reasons.append(
            "reward_command_received_count=%d planned_reward_count=%d" % (len(reward_command_received_events), planned_reward_count)
        )
    if len(reward_complete_events) != planned_reward_count:
        qc_fail_reasons.append(
            "reward_complete_count=%d planned_reward_count=%d" % (len(reward_complete_events), planned_reward_count)
        )
    if len(reward_valve_on_events) != planned_reward_count * int(reward_num_pulses):
        qc_fail_reasons.append(
            "reward_valve_on_count=%d expected_valve_on_count=%d" % (len(reward_valve_on_events), planned_reward_count * int(reward_num_pulses))
        )
    if len(reward_valve_off_events) != planned_reward_count * int(reward_num_pulses):
        qc_fail_reasons.append(
            "reward_valve_off_count=%d expected_valve_off_count=%d" % (len(reward_valve_off_events), planned_reward_count * int(reward_num_pulses))
        )
    if missing_reward_command_ids:
        qc_fail_reasons.append("missing_reward_command_ids=%s" % ",".join(missing_reward_command_ids))
    if unexpected_reward_command_ids:
        qc_fail_reasons.append("unexpected_reward_command_ids=%s" % ",".join(unexpected_reward_command_ids))
    if incomplete_reward_command_ids:
        qc_fail_reasons.append("incomplete_reward_command_ids=%s" % ",".join(incomplete_reward_command_ids))
    if duplicate_reward_command_ids:
        qc_fail_reasons.append("duplicate_reward_command_ids=%s" % ",".join(duplicate_reward_command_ids))
    if len(suction_command_received_events) != planned_suction_count:
        qc_fail_reasons.append("suction_command_received_count=%d planned_suction_count=%d" % (len(suction_command_received_events), planned_suction_count))
    if len(suction_complete_events) != planned_suction_count:
        qc_fail_reasons.append("suction_complete_count=%d planned_suction_count=%d" % (len(suction_complete_events), planned_suction_count))
    if unexpected_suction_command_ids:
        qc_fail_reasons.append(
            "unexpected_suction_command_ids=%s" % ",".join(unexpected_suction_command_ids)
        )
    if missing_suction_command_ids or duplicate_suction_command_ids or incomplete_suction_command_ids:
        qc_fail_reasons.append("suction_command_integrity_failure")

    qc_pass = not qc_fail_reasons
    reward_volume_summary = dict(reward_volume_summary or {})
    return {
        "schema_version": QC_SCHEMA_VERSION,
        "session_id": session_id,
        "generated_utc_iso": base.utc_iso_now(),
        "planned_trial_count": planned_trial_count,
        "executed_trial_count": executed_trial_count,
        "completed_trial_count": completed_trial_count,
        "planned_reward_count": planned_reward_count,
        "planned_suction_count": planned_suction_count,
        "camera_data_secured": camera_data_secured,
        "reward_volume_cap_enabled": reward_volume_summary.get("reward_volume_cap_enabled", False),
        "reward_volume_ul_per_train": reward_volume_summary.get("reward_volume_ul_per_train"),
        "maximum_session_reward_ul": reward_volume_summary.get("maximum_session_reward_ul"),
        "planned_reward_volume_ul": reward_volume_summary.get("planned_reward_volume_ul"),
        "manual_reward_train_count": reward_volume_summary.get("manual_reward_train_count", 0),
        "delivered_reward_train_count": reward_volume_summary.get("delivered_reward_train_count", 0),
        "estimated_delivered_reward_ul": reward_volume_summary.get("estimated_delivered_reward_ul"),
        "reward_volume_cap_exceeded": reward_volume_summary.get("reward_volume_cap_exceeded", False),
        "reward_command_received_count": len(reward_command_received_events),
        "reward_complete_count": len(reward_complete_events),
        "reward_valve_on_count": len(reward_valve_on_events),
        "reward_valve_off_count": len(reward_valve_off_events),
        "expected_valve_on_count": planned_reward_count * int(reward_num_pulses),
        "expected_valve_off_count": planned_reward_count * int(reward_num_pulses),
        "missing_reward_command_ids": missing_reward_command_ids,
        "unexpected_reward_command_ids": unexpected_reward_command_ids,
        "incomplete_reward_command_ids": incomplete_reward_command_ids,
        "duplicate_reward_command_ids": duplicate_reward_command_ids,
        "suction_command_received_count": len(suction_command_received_events),
        "suction_complete_count": len(suction_complete_events),
        "suction_on_count": len(suction_on_events),
        "suction_off_count": len(suction_off_events),
        "missing_suction_command_ids": missing_suction_command_ids,
        "unexpected_suction_command_ids": unexpected_suction_command_ids,
        "duplicate_suction_command_ids": duplicate_suction_command_ids,
        "incomplete_suction_command_ids": incomplete_suction_command_ids,
        "lick_onset_count": len(lick_onset_events),
        "segment_boundary_gap_sec_median": _series_stat(segment_boundary_gap_values, "segment_boundary_gap_sec", "median"),
        "segment_boundary_gap_sec_p95": _series_stat(segment_boundary_gap_values, "segment_boundary_gap_sec", "p95"),
        "segment_boundary_gap_sec_max": _series_stat(segment_boundary_gap_values, "segment_boundary_gap_sec", "max"),
        "software_reward_delay_sec_median": _series_stat(software_reward_delay_values, "software_reward_delay_sec", "median"),
        "software_reward_delay_sec_p95": _series_stat(software_reward_delay_values, "software_reward_delay_sec", "p95"),
        "software_reward_delay_sec_max": _series_stat(software_reward_delay_values, "software_reward_delay_sec", "max"),
        "software_suction_delay_sec_median": _series_stat([row["software_suction_delay_sec"] for row in trial_summary_rows if row.get("software_suction_delay_sec") not in ("", None)], "software_suction_delay_sec", "median"),
        "software_suction_delay_sec_p95": _series_stat([row["software_suction_delay_sec"] for row in trial_summary_rows if row.get("software_suction_delay_sec") not in ("", None)], "software_suction_delay_sec", "p95"),
        "software_suction_delay_sec_max": _series_stat([row["software_suction_delay_sec"] for row in trial_summary_rows if row.get("software_suction_delay_sec") not in ("", None)], "software_suction_delay_sec", "max"),
        "qc_pass": qc_pass,
        "qc_fail_reasons": qc_fail_reasons,
    }


def maybe_import_camera_support(no_camera):
    if no_camera:
        return None
    try:
        import run_stringer_vstim_cam as camera_support

        return camera_support
    except Exception as exc:
        print("Remote camera support unavailable: %s" % exc)
        return None


def format_setup_summary(mouse_id, session_notes, assignment_path,
                         assignment_created, n_blocks, trials, iti_min_sec,
                         iti_max_sec, pre_sec, post_sec, planned_task_sec,
                         planned_total_sec, hardware_config, use_camera,
                         suction_delay_sec, reward_train_duration_sec,
                         reward_timing=None,
                         output_root=OUTPUT_ROOT,
                         display_timing=None):
    if reward_timing is None:
        reward_extension_sec = max(
            0.0, reward_train_duration_sec - POST_REWARD_STIM_SEC
        )
        reward_timing = {
            "reward_nominal_completion_from_stim_onset_sec": (
                REWARD_DELAY_SEC + reward_train_duration_sec
            ),
            "reward_train_extension_into_iti_sec": reward_extension_sec,
            "reward_train_extends_into_iti": reward_extension_sec > 0.0,
            "reward_to_suction_nominal_gap_sec": (
                suction_delay_sec
                - REWARD_DELAY_SEC
                - reward_train_duration_sec
            ),
        }
    if display_timing is None:
        display_timing = resolve_display_timing_calibration({})
    calibration_label = display_timing["calibration_id"] or "none (nominal fallback)"
    lines = [
        "Session setup summary:",
        "  Mouse: %s" % mouse_id,
        "  Session notes: %s" % (session_notes or "(none)"),
        "  Shared 14-image panel: %s" % global_panel_path(ASSIGNMENT_DIR),
        "  Fixed per-mouse role assignment: %s" % assignment_path,
        "  Assignment newly created: %s" % assignment_created,
        "  Blocks: %d x %d trials" % (n_blocks, TRIALS_PER_BLOCK),
        "  Total trials: %d" % len(trials),
        "  Scheduled rewards: %d" % sum(1 for trial in trials if trial["reward_scheduled"]),
        "  Scheduled omissions: %d" % sum(1 for trial in trials if trial["reward_omission_scheduled"]),
        "  Scheduled suction events: %d" % sum(1 for trial in trials if trial.get("suction_scheduled")),
        "  Stimulus: 1.5 s; open-loop reward boundary: 1.0 s",
        "  Display timing calibration: %s" % calibration_label,
        "  Display refresh profile: %.1f Hz; segments %d/%d refreshes (%.6f/%.6f s programmed)"
        % (
            display_timing["calibration_refresh_rate_hz"] or base.REFRESH_RATE_HZ,
            display_timing["segment1_refreshes"],
            display_timing["segment2_refreshes"],
            display_timing["segment1_programmed_duration_sec"],
            display_timing["segment2_programmed_duration_sec"],
        ),
        "  Total programmed stimulus: %d refreshes (%.6f s); suction onset compensation: +%.3f s"
        % (
            display_timing["stim_total_refreshes"],
            display_timing["total_programmed_duration_sec"],
            display_timing["stimulus_onset_compensation_sec"],
        ),
        "  Reward probability: %.3f (fixed)" % REWARD_PROBABILITY,
        "  ITI: uniform %.3f-%.3f s" % (iti_min_sec, iti_max_sec),
        "  PRE gray background: %s" % base.format_seconds(pre_sec),
        "  POST gray background: %s" % base.format_seconds(post_sec),
        "  Planned task duration: %s" % base.format_seconds(planned_task_sec),
        "  Planned PRE + task + POST: %s" % base.format_seconds(planned_total_sec),
        "  Camera/video cleanup not included.",
        "  Reward pin: BCM%d" % hardware_config["reward_pin_bcm"],
        "  Lick pin: BCM%d" % hardware_config["lick_pin_bcm"],
        "  Suction pin: BCM%d" % hardware_config["suction_pin_bcm"],
        "  Reward pulse-train duration: %.3f s" % reward_train_duration_sec,
        "  Reward nominal completion: %.3f s after stimulus onset"
        % reward_timing["reward_nominal_completion_from_stim_onset_sec"],
        "  Reward extends into gray ITI: %s (%.3f s)"
        % (
            "yes" if reward_timing["reward_train_extends_into_iti"] else "no",
            reward_timing["reward_train_extension_into_iti_sec"],
        ),
        "  Suction onset: %.3f s after stimulus onset" % suction_delay_sec,
        "  Reward-to-suction nominal gap: %.3f s"
        % reward_timing["reward_to_suction_nominal_gap_sec"],
        "  GPIO simulation: %s" % hardware_config["simulate_gpio"],
        "  Face camera: %s" % use_camera,
        "  Output destination: %s/<session_id>" % output_root,
        "  IMPORTANT: reward delivery is independent of licking.",
        "  Suction is applied to rewarded and omission conditioned-cue trials, independent of licking.",
    ]
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    base.print_environment()
    hardware_config = load_hardware_config(
        args.hardware_config, simulate_gpio=args.simulate_gpio
    )
    suction_delay_sec = float(hardware_config["suction_delay_from_stim_onset_sec"])
    reward_timing = calculate_reward_timing(hardware_config)
    display_timing = resolve_display_timing_calibration(hardware_config)
    reward_train_duration_sec = reward_timing["reward_train_duration_sec"]
    reward_verification_timeout_sec = reward_train_duration_sec + REWARD_VERIFICATION_MARGIN_SEC
    try:
        import rpg
    except ImportError as exc:
        raise RuntimeError(
            "The rpg package is not installed. Install the SjulsonLab rpg repo "
            "on the behavior Pi first."
        ) from exc

    camera_support = maybe_import_camera_support(args.no_camera)
    if args.camera and camera_support is None:
        raise RuntimeError("--camera was requested but camera support is unavailable.")

    if args.mouse_id is not None:
        mouse_id_raw = args.mouse_id
        mouse_id = base.sanitize_text(mouse_id_raw)
        if not mouse_id: raise ValueError("Mouse ID is required after sanitization.")
    else:
        while True:
            mouse_id_raw = _strict_prompt("Mouse ID: ")
            mouse_id = base.sanitize_text(mouse_id_raw)
            if mouse_id: break
            print("Mouse ID is required; try again.")
    session_notes = args.session_notes if args.session_notes is not None else _strict_prompt("Session notes, optional: ")
    n_blocks = args.blocks if args.blocks is not None else prompt_int_with_default("Number of 50-trial probability blocks", DEFAULT_N_BLOCKS, minimum=1)
    if int(n_blocks) < 1: raise ValueError("blocks must be at least 1")
    iti_min_sec = args.iti_min_sec if args.iti_min_sec is not None else prompt_float_or_default(
        "Minimum gray ITI in seconds",
        DEFAULT_ITI_MIN_SEC,
        minimum=0.1 if args.simulate_gpio else DEFAULT_ITI_MIN_SEC,
    )
    iti_max_sec = args.iti_max_sec if args.iti_max_sec is not None else prompt_float_or_default(
        "Maximum gray ITI in seconds", DEFAULT_ITI_MAX_SEC, minimum=iti_min_sec
    )
    pre_background_min = args.pre_background_min if args.pre_background_min is not None else prompt_float_or_default(
        "PRE gray-background duration in minutes",
        DEFAULT_PRE_BACKGROUND_MIN,
        minimum=0.0,
    )
    post_background_min = args.post_background_min if args.post_background_min is not None else prompt_float_or_default(
        "POST gray-background duration in minutes",
        DEFAULT_POST_BACKGROUND_MIN,
        minimum=0.0,
    )
    for label, value, minimum in (("iti_min_sec", iti_min_sec, 0.1), ("iti_max_sec", iti_max_sec, iti_min_sec),
                                  ("pre_background_min", pre_background_min, 0.0), ("post_background_min", post_background_min, 0.0)):
        if not math.isfinite(float(value)) or float(value) < minimum:
            raise ValueError("%s is invalid" % label)
    use_camera = resolve_camera_choice(
        args.camera, args.no_camera, camera_support,
        prompt_fn=prompt_yes_no_strict)

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
        suction_delay_sec=hardware_config["suction_delay_from_stim_onset_sec"],
    )
    reward_volume_tracker = RewardVolumeTracker(
        hardware_config.get("reward_volume_ul_per_train"),
        hardware_config.get("maximum_session_reward_ul"),
        planned_reward_train_count(trials),
    )
    try:
        reward_volume_tracker.preflight()
    except RewardVolumeCapExceeded as exc:
        print("REWARD SAFETY ERROR: %s" % exc, file=sys.stderr)
        raise
    plan_summary = summarize_trial_plan(trials)
    planned_task_sec = estimate_task_seconds(trials)
    planned_total_sec = (
        pre_background_min * 60.0
        + planned_task_sec
        + post_background_min * 60.0
    )

    print()
    print(format_setup_summary(
        mouse_id, session_notes, assignment_path, assignment_created,
        n_blocks, trials, iti_min_sec, iti_max_sec,
        pre_background_min * 60.0, post_background_min * 60.0,
        planned_task_sec, planned_total_sec, hardware_config, use_camera,
        suction_delay_sec, reward_train_duration_sec,
        reward_timing=reward_timing, display_timing=display_timing))
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
    if not prompt_yes_no_strict("Create files and prepare this session", default_yes=True):
        print("Session aborted before hardware start.")
        return 0

    session_stamp = base.utc_session_stamp()
    session_id = make_session_name(mouse_id, session_stamp)
    session_root = base.ensure_dir(OUTPUT_ROOT / session_id)
    raw_cache_root = base.ensure_dir(session_root / "raw_cache")
    event_log_path = session_root / (session_id + "_event_log.csv")
    camera_event_log_path = session_root / "video" / CAMERA_EVENT_LOG_FILENAME
    planned_sequence_path = session_root / (session_id + "_planned_sequence.csv")
    selected_images_path = session_root / (session_id + "_image_assignment.csv")
    plan_summary_path = session_root / (session_id + "_plan_summary.csv")
    trial_summary_path = session_root / (session_id + "_trial_summary.csv")
    lick_events_path = session_root / (session_id + "_lick_events.csv")
    qc_path = session_root / (session_id + "_session_qc.json")
    metadata_path = session_root / (session_id + "_metadata.json")
    manifest_path = session_root / "session_manifest.json"

    telemetry_publisher = TelemetryPublisher(
        host=args.telemetry_host,
        port=args.telemetry_port,
        session_id=session_id,
        protocol="reward_conditioning",
        enabled=not args.no_telemetry,
    )
    telemetry_started = telemetry_publisher.start()
    if not args.no_telemetry and not telemetry_started:
        print(
            "Telemetry unavailable (%s); continuing without monitor updates."
            % (telemetry_publisher.start_error or "startup failed"),
            file=sys.stderr,
        )
    telemetry_state_reporter = TelemetryStateReporter(telemetry_publisher)
    water_accounting = TaskWaterTelemetryAccounting(
        reward_volume_tracker.reward_volume_ul_per_train
    )
    telemetry_publisher.publish_session({
        "phase": "PREPARING",
        "total_trials": len(trials),
        "total_blocks": int(n_blocks),
        **_telemetry_water_fields(water_accounting),
    })

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
        "display_timing_calibration_enabled": display_timing["calibration_enabled"],
        "display_timing_calibration_id": display_timing["calibration_id"],
        "display_timing_calibration_refresh_rate_hz": display_timing["calibration_refresh_rate_hz"],
        "stim_segment1_refreshes": display_timing["segment1_refreshes"],
        "stim_segment2_refreshes": display_timing["segment2_refreshes"],
        "stim_total_refreshes": display_timing["stim_total_refreshes"],
        "stim_segment1_programmed_duration_sec": display_timing["segment1_programmed_duration_sec"],
        "stim_segment2_programmed_duration_sec": display_timing["segment2_programmed_duration_sec"],
        "stim_total_programmed_duration_sec": display_timing["total_programmed_duration_sec"],
        "stimulus_onset_compensation_sec": display_timing["stimulus_onset_compensation_sec"],
        "behavioral_target_reward_delay_sec": REWARD_DELAY_SEC,
        "behavioral_target_stim_duration_sec": STIM_DURATION_SEC,
        "behavioral_target_suction_delay_sec": suction_delay_sec,
        "telemetry_enabled": bool(telemetry_started),
        "telemetry_host": args.telemetry_host,
        "telemetry_port": args.telemetry_port,
        "telemetry_queue_dropped_count": telemetry_publisher.dropped_count,
        "telemetry_parent_error_count": telemetry_publisher.parent_error_count,
        "telemetry_start_error": telemetry_publisher.start_error or "",
        **_telemetry_water_fields(water_accounting),
        "iti_distribution": "uniform",
        "iti_min_sec": iti_min_sec,
        "iti_max_sec": iti_max_sec,
        "pre_background_requested_sec": pre_background_min * 60.0,
        "post_background_requested_sec": post_background_min * 60.0,
        "session_output_schema_version": SESSION_OUTPUT_SCHEMA_VERSION,
        "event_log_schema_version": EVENT_LOG_SCHEMA_VERSION,
        "trial_summary_schema_version": TRIAL_SUMMARY_SCHEMA_VERSION,
        "qc_schema_version": QC_SCHEMA_VERSION,
        "planned_task_duration_sec": planned_task_sec,
        "planned_pre_duration_sec": pre_background_min * 60.0,
        "planned_post_duration_sec": post_background_min * 60.0,
        "planned_protocol_duration_sec": planned_total_sec,
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
        "reward_train_extends_into_iti": reward_timing["reward_train_extends_into_iti"],
        "reward_train_extension_into_iti_sec": reward_timing["reward_train_extension_into_iti_sec"],
        "reward_nominal_start_from_stim_onset_sec": reward_timing["reward_nominal_start_from_stim_onset_sec"],
        "reward_nominal_completion_from_stim_onset_sec": reward_timing["reward_nominal_completion_from_stim_onset_sec"],
        "reward_to_suction_nominal_gap_sec": reward_timing["reward_to_suction_nominal_gap_sec"],
        **reward_volume_tracker.summary(),
        "base_ttl_bcm23_used": False,
        "face_camera_requested": use_camera,
        "camera_data_secured": not use_camera,
        "screen_resolution": list(base.SCREEN_RESOLUTION),
        "screen_background_gray": base.SCREEN_BACKGROUND_GRAY,
        "refresh_rate_hz": base.REFRESH_RATE_HZ,
        "photodiode_patch_enabled": base.ENABLE_PHOTODIODE_PATCH,
        "photodiode_size_px": base.PHOTODIODE_SIZE_PX,
        "vstim_natural_img_reward_git_commit": get_git_commit(),
        "vstim_natural_git_commit": get_git_commit(),
        "planned_sequence_csv": str(planned_sequence_path),
        "event_log_csv": str(event_log_path),
        "trial_summary_csv": str(trial_summary_path),
        "camera_event_log_csv": str(camera_event_log_path) if use_camera else "",
        "camera_recording_elapsed_local_sec": None,
        "camera_startup_confirmation_latency_sec": None,
        "session_qc_json": "",
        "lick_events_csv": str(lick_events_path),
        "suction_delay_from_stim_onset_sec": suction_delay_sec,
        "final_planned_iti_executed": False,
        "post_started_immediately_after_final_stimulus": True,
        "plan_summary": plan_summary,
    }
    atomic_write_json(metadata_path, metadata)
    atomic_write_json(
        manifest_path,
        build_session_manifest(
            session_root, session_id, mouse_id, metadata["protocol"],
            "preparing", {
                "metadata": metadata_path,
                "planned_sequence": planned_sequence_path,
                "image_assignment": selected_images_path,
                "plan_summary": plan_summary_path,
                "event_log": event_log_path,
                "trial_summary": trial_summary_path,
                "lick_events": lick_events_path,
                "session_qc": qc_path,
                "camera_event_log": camera_event_log_path if use_camera else None,
                "video_manifest": (
                    session_root / "video" / "video_manifest.json"
                    if use_camera else None),
            },
        ),
    )

    all_gpio_events = []
    runtime_by_trial = {}
    gpio_client = BehaviorGPIOClient(hardware_config)
    task_completed = False
    post_background_completed = False
    camera_started = False
    camera_stop_confirmed = False
    camera_fetch_completed = False
    camera_conversion_completed = False
    camera_conversion_deferred = False
    camera_cleanup_error = False
    camera_cleanup_error_message = ""
    session_completed = False
    pending_final_reward_checks = []
    interrupted = False
    primary_error = None
    cleanup_errors = []
    finalization_errors = []
    latest_camera_state = {}
    status_reporter = StatusReporter()

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
                rpg, raw_cache_root, assignment_rows, display_timing
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

            if prompt_yes_no_strict(
                "Deliver one manual test reward before starting baselines",
                default_yes=False,
            ):
                try:
                    reward_volume_tracker.check_next_command(scheduled=False)
                    command_id = gpio_client.manual_reward()
                    reward_volume_tracker.record_dispatched(manual=True)
                except RewardVolumeCapExceeded as exc:
                    _record_reward_volume_block(
                        event_log_path,
                        {"phase": "manual_reward", "reward_trigger_source": "manual_test"},
                        exc,
                    )
                    raise
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

            if prompt_yes_no_strict(
                "Activate one manual suction pulse before starting baselines",
                default_yes=False,
            ):
                command_id = gpio_client.manual_suction()
                append_event(event_log_path, exact_timestamp_event(
                    "manual_suction_command_sent", phase="manual_suction",
                    command_id=command_id, suction_pin_bcm=hardware_config["suction_pin_bcm"],
                    notes="operator_requested_test_suction"))
                wait_until(time.monotonic() + hardware_config["suction_duration_sec"] + 0.25,
                           gpio_client, event_log_path, all_gpio_events)

            if use_camera:
                metadata["camera_timer_anchor"] = "local_monotonic_at_camera_recording_confirmed"
                metadata["camera_recording_request_monotonic_ns"] = time.monotonic_ns()
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
                metadata["camera_recording_confirmed_monotonic_ns"] = time.monotonic_ns()
                metadata["camera_startup_confirmation_latency_sec"] = (
                    camera_startup_confirmation_latency_from_ns(
                        metadata["camera_recording_request_monotonic_ns"],
                        metadata["camera_recording_confirmed_monotonic_ns"],
                    )
                )
                metadata["camera_start_result"] = camera_result
                latest_camera_state = dict(camera_result.get("controller_state", {}))
                append_event(
                    event_log_path,
                    exact_timestamp_event(
                        "camera_recording_confirmed",
                        phase="preparation_gray",
                        notes="remote_camera_output_growing_confirmed",
                    ),
                )

            metadata["operator_gate_enter_monotonic_ns"] = time.monotonic_ns()
            append_event(event_log_path, exact_timestamp_event("two_photon_operator_gate_entered", phase="preparation_gray"))
            gate_camera_anchor = metadata.get("camera_recording_confirmed_monotonic_ns")

            def waiting_status_callback():
                status_reporter.report(format_operator_status("WAITING_FOR_2P",
                    (time.monotonic_ns() - gate_camera_anchor) / 1e9 if gate_camera_anchor else None,
                    planned_total_sec))
                safe_report_telemetry_state(
                    telemetry_state_reporter, "WAITING_FOR_2P",
                    total_trials=len(trials), total_blocks=n_blocks,
                    eta_sec=planned_total_sec,
                    water_accounting=water_accounting,
                )

            gate_wait_sec = wait_for_two_photon_gate(
                waiting_status_callback,
                service_callback=lambda: log_drained_gpio_events(gpio_client, event_log_path, all_gpio_events))
            status_reporter.finalize()
            metadata["operator_gate_release_monotonic_ns"] = time.monotonic_ns()
            metadata["operator_gate_wait_sec"] = gate_wait_sec
            append_event(
                event_log_path,
                exact_timestamp_event(
                    "two_photon_operator_gate_released",
                    phase="preparation_gray",
                    notes="operator_pressed_enter",
                ),
            )

            metadata["pre_start_monotonic_ns"] = time.monotonic_ns()
            safe_report_telemetry_state(
                telemetry_state_reporter, "PRE",
                total_trials=len(trials), total_blocks=n_blocks,
                eta_sec=pre_background_min * 60.0 + planned_task_sec
                + post_background_min * 60.0,
                water_accounting=water_accounting,
                force=True,
            )

            def pre_status_callback(remaining):
                status_reporter.report(format_operator_status("PRE",
                    (time.monotonic_ns() - gate_camera_anchor) / 1e9 if gate_camera_anchor else None,
                    remaining,
                    protocol_remaining_sec=(remaining + planned_task_sec + post_background_min * 60.0)))
                safe_report_telemetry_state(
                    telemetry_state_reporter, "PRE",
                    total_trials=len(trials), total_blocks=n_blocks,
                    eta_sec=remaining + planned_task_sec + post_background_min * 60.0,
                    water_accounting=water_accounting,
                )

            pre_actual = hold_background(
                "prestim_background",
                pre_background_min * 60.0,
                gpio_client,
                event_log_path,
                all_gpio_events,
                status_callback=pre_status_callback,
            )
            metadata["pre_end_monotonic_ns"] = time.monotonic_ns()
            metadata["pre_elapsed_sec"] = pre_actual
            metadata["pre_background_actual_sec"] = pre_actual

            append_event(
                event_log_path,
                exact_timestamp_event(
                    "task_start",
                    phase="task",
                    notes="first stimulus follows; reward is open-loop",
                ),
            )
            metadata["task_start_monotonic_ns"] = time.monotonic_ns()

            def task_status_callback(done, total, remaining):
                status_reporter.report(format_operator_status("TASK",
                    (time.monotonic_ns() - gate_camera_anchor) / 1e9 if gate_camera_anchor else None,
                    remaining, done, total, post_background_min * 60.0))

            runtime_by_trial, pending_final_reward_checks = run_trials(
                screen,
                trials,
                loaded_raws,
                loaded_background_raw,
                raw_paths,
                background_raw_path,
                gpio_client,
                event_log_path,
                all_gpio_events,
                hardware_config["reward_num_pulses"],
                reward_verification_timeout_sec,
                hardware_config["suction_delay_from_stim_onset_sec"],
                hardware_config["suction_duration_sec"],
                stimulus_onset_compensation_sec=display_timing["stimulus_onset_compensation_sec"],
                status_callback=task_status_callback,
                reward_volume_tracker=reward_volume_tracker,
                telemetry_publisher=telemetry_publisher,
                telemetry_state_reporter=telemetry_state_reporter,
                water_accounting=water_accounting,
                total_blocks=n_blocks,
                post_background_sec=post_background_min * 60.0,
            )
            metadata["task_end_monotonic_ns"] = time.monotonic_ns()
            metadata["task_elapsed_sec"] = (metadata["task_end_monotonic_ns"] - metadata["task_start_monotonic_ns"]) / 1e9
            task_completed = True
            append_event(
                event_log_path,
                exact_timestamp_event(
                    "task_end",
                    phase="task",
                    notes="all planned trials completed",
                ),
            )

            metadata["post_start_monotonic_ns"] = time.monotonic_ns()
            safe_report_telemetry_state(
                telemetry_state_reporter, "POST",
                total_trials=len(trials), total_blocks=n_blocks,
                eta_sec=post_background_min * 60.0,
                water_accounting=water_accounting,
                force=True,
            )

            def post_status_callback(remaining):
                status_reporter.report(format_operator_status("POST",
                    (time.monotonic_ns() - gate_camera_anchor) / 1e9 if gate_camera_anchor else None,
                    remaining))
                safe_report_telemetry_state(
                    telemetry_state_reporter, "POST",
                    total_trials=len(trials), total_blocks=n_blocks,
                    eta_sec=remaining, water_accounting=water_accounting,
                )

            def final_trial_telemetry_callback():
                if not trials:
                    return
                final_trial = trials[-1]
                final_runtime = runtime_by_trial.get(final_trial["trial_index"])
                if final_runtime is None:
                    return
                publish_completed_trial_telemetry(
                    telemetry_publisher,
                    final_trial,
                    final_runtime,
                    all_gpio_events,
                    len(trials),
                    n_blocks,
                    hardware_config["reward_num_pulses"],
                    display_timing["stimulus_onset_compensation_sec"],
                    post_background_min * 60.0,
                    water_accounting,
                )

            post_actual = hold_background(
                "poststim_background",
                post_background_min * 60.0,
                gpio_client,
                event_log_path,
                all_gpio_events,
                pending_reward_checks=pending_final_reward_checks,
                status_callback=post_status_callback,
                on_pending_checks_complete=final_trial_telemetry_callback,
            )
            metadata["post_end_monotonic_ns"] = time.monotonic_ns()
            metadata["post_elapsed_sec"] = post_actual
            metadata["post_background_actual_sec"] = post_actual
            post_background_completed = True
            safe_report_telemetry_state(
                telemetry_state_reporter, "COMPLETE",
                total_trials=len(trials), total_blocks=n_blocks,
                eta_sec=0.0, water_accounting=water_accounting,
                force=True,
            )

            # Keep exactly the same gray background on screen during remote
            # camera cleanup.  The current natural-stim camera runner switches
            # to black; this task intentionally does not.
            if use_camera and camera_started:
                try:
                    stop_state = camera_support.stop_camera_recording()
                    camera_stop_confirmed = bool(stop_state.get("camera_stop_confirmed", False))
                    latest_camera_state = dict(stop_state)
                    if not camera_stop_confirmed:
                        raise RuntimeError("Camera stop was not confirmed: %s" % json.dumps(stop_state, sort_keys=True))
                    metadata["camera_stop_confirmed_monotonic_ns"] = time.monotonic_ns()
                    anchor_ns = metadata.get("camera_recording_confirmed_monotonic_ns")
                    metadata["camera_recording_elapsed_local_sec"] = camera_elapsed_from_ns(
                        anchor_ns, metadata["camera_stop_confirmed_monotonic_ns"])
                    status_reporter.finalize("REC stopped %s | camera stop confirmed" % base.format_seconds(metadata["camera_recording_elapsed_local_sec"] or 0))
                    append_event(
                        event_log_path,
                        exact_timestamp_event(
                            "camera_stop_confirmed",
                            phase="poststim_camera_cleanup_gray",
                            notes="screen_remained_gray",
                        ),
                    )
                    print("VIDEO | verifying/fetching raw recording...")
                    fetch_state = camera_support.fetch_camera_recording()
                    metadata["camera_fetch_result"] = fetch_state
                    latest_camera_state = dict(fetch_state)
                    camera_fetch_completed = bool(
                        fetch_state.get("camera_fetch_completed", False)
                    )
                    append_event(
                        event_log_path,
                        exact_timestamp_event(
                            "camera_fetch_returned",
                            phase="poststim_camera_cleanup_gray",
                            notes=json.dumps(fetch_state, sort_keys=True),
                        ),
                    )
                    camera_conversion_completed = bool(
                        fetch_state.get("camera_conversion_completed", False)
                    )
                    camera_conversion_deferred = bool(
                        fetch_state.get("camera_conversion_deferred", False)
                    )
                    append_event(
                        event_log_path,
                        exact_timestamp_event(
                            "camera_conversion_returned",
                            phase="poststim_camera_cleanup_gray",
                            notes=json.dumps(fetch_state, sort_keys=True),
                        ),
                    )
                    if camera_fetch_completed and not camera_conversion_completed:
                        print("VIDEO | converting/verifying MP4...")
                        convert_state = camera_support.convert_camera_recording()
                        metadata["camera_convert_result"] = convert_state
                        latest_camera_state = dict(convert_state)
                        camera_conversion_completed = bool(
                            convert_state.get("camera_conversion_completed", False)
                        )
                        append_event(
                            event_log_path,
                            exact_timestamp_event(
                                "camera_conversion_retry_returned",
                                phase="poststim_camera_cleanup_gray",
                                notes=json.dumps(convert_state, sort_keys=True),
                            ),
                        )
                except Exception as exc:
                    camera_cleanup_error = True
                    camera_cleanup_error_message = "%s: %s" % (type(exc).__name__, exc)
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
        interrupted = True
        try:
            append_event(event_log_path, exact_timestamp_event(
                "session_interrupted", phase="unknown", notes="KeyboardInterrupt"))
        except Exception as exc:
            cleanup_errors.append("interrupt_log: %s: %s" % (type(exc).__name__, exc))
        print("Session interrupted by Ctrl-C; cleaning up hardware.", file=sys.stderr)
    except Exception as exc:
        primary_error = exc
        try:
            append_event(event_log_path, exact_timestamp_event(
                "session_error", phase="unknown", notes="%s: %s" % (type(exc).__name__, exc)))
        except Exception as log_exc:
            cleanup_errors.append("error_log: %s: %s" % (type(log_exc).__name__, log_exc))
    finally:
        if camera_started and not camera_stop_confirmed and camera_support is not None:
            try:
                stop_state = camera_support.stop_camera_recording()
                camera_stop_confirmed = bool(stop_state.get("camera_stop_confirmed", False))
                latest_camera_state = dict(stop_state)
                if camera_stop_confirmed and "camera_stop_confirmed_monotonic_ns" not in metadata:
                    metadata["camera_stop_confirmed_monotonic_ns"] = time.monotonic_ns()
                    anchor_ns = metadata.get("camera_recording_confirmed_monotonic_ns")
                    metadata["camera_recording_elapsed_local_sec"] = camera_elapsed_from_ns(
                        anchor_ns, metadata["camera_stop_confirmed_monotonic_ns"])
                append_event(
                    event_log_path,
                    exact_timestamp_event(
                        "camera_stop_returned",
                        phase="poststim_camera_cleanup_gray",
                        notes=json.dumps(stop_state, sort_keys=True),
                    ),
                )
            except Exception as exc:
                camera_cleanup_error = True
                camera_cleanup_error_message = "%s: %s" % (type(exc).__name__, exc)
                print("Emergency camera stop failed: %s" % exc, file=sys.stderr)
                cleanup_errors.append("camera_stop: %s: %s" % (type(exc).__name__, exc))

        if gpio_client is not None:
            try:
                shutdown_events = gpio_client.shutdown()
                for event in shutdown_events:
                    append_event(event_log_path, event)
                    if event.get("message_type") == "event":
                        all_gpio_events.append(dict(event))
            except Exception as exc:
                print("GPIO shutdown error: %s" % exc, file=sys.stderr)
                cleanup_errors.append("gpio_shutdown: %s: %s" % (type(exc).__name__, exc))

        if (use_camera and latest_camera_state.get("remote_raw_cleanup_error")
                and not camera_cleanup_error):
            cleanup_errors.append(
                "camera_cleanup: %s"
                % latest_camera_state.get("remote_raw_cleanup_error"))
        preliminary_completion = resolve_final_completion_state(
            task_completed, post_background_completed, interrupted,
            primary_error, use_camera, latest_camera_state,
            camera_cleanup_error, cleanup_errors, finalization_errors)
        camera_data_secured = preliminary_completion["camera_data_secured"]
        try:
            trial_summary = build_trial_summary(
                trials,
                runtime_by_trial,
                all_gpio_events,
                stimulus_onset_compensation_sec=display_timing["stimulus_onset_compensation_sec"],
            )
            write_rows(trial_summary_path, trial_summary, TRIAL_SUMMARY_FIELDS)
            lick_rows = [event for event in all_gpio_events if event.get("event_type") in ("lick_onset", "lick_offset")]
            write_rows(lick_events_path, lick_rows, ["unix_time_ns", "monotonic_ns", "event_type", "phase", "trial_index", "trial_number", "block_number", "image_role", "image_filename", "reward_scheduled", "suction_scheduled"])
            qc = build_session_qc(
                session_id, trials, trial_summary, all_gpio_events,
                hardware_config["reward_num_pulses"],
                reward_volume_summary=reward_volume_tracker.summary(),
                camera_data_secured=camera_data_secured,
            )
            atomic_write_json(qc_path, qc)
            metadata["session_qc_json"] = str(qc_path)
        except Exception as exc:
            finalization_errors.append("session_artifacts: %s: %s" % (type(exc).__name__, exc))

        metadata["utc_iso_end"] = base.utc_iso_now()
        metadata["task_completed"] = task_completed
        metadata["post_background_completed"] = post_background_completed
        metadata["camera_data_secured"] = camera_data_secured
        metadata.update(reward_volume_tracker.summary())
        metadata.update(_telemetry_water_fields(water_accounting))
        metadata["telemetry_queue_dropped_count"] = telemetry_publisher.dropped_count
        metadata["telemetry_parent_error_count"] = telemetry_publisher.parent_error_count
        if task_completed and post_background_completed:
            try:
                validate_operational_timing_metadata(
                    metadata,
                    camera_required=bool(
                        use_camera and camera_started and camera_stop_confirmed),
                )
            except Exception as exc:
                finalization_errors.append(
                    "operational_timing_metadata: %s: %s"
                    % (type(exc).__name__, exc))
        metadata.update(final_camera_metadata(
            latest_camera_state, use_camera, camera_started, camera_stop_confirmed,
            camera_fetch_completed, camera_conversion_completed))
        metadata["camera_conversion_deferred"] = camera_conversion_deferred
        metadata["camera_cleanup_error"] = camera_cleanup_error
        metadata["camera_cleanup_error_message"] = camera_cleanup_error_message
        metadata["camera_requested"] = bool(use_camera)
        metadata["cleanup_completed"] = not cleanup_errors
        metadata["interrupted"] = interrupted
        metadata["primary_error"] = "" if primary_error is None else "%s: %s" % (type(primary_error).__name__, primary_error)
        metadata["cleanup_errors"] = cleanup_errors
        metadata["finalization_errors"] = finalization_errors
        metadata["n_gpio_events_logged"] = len(all_gpio_events)
        completion_state = finalize_status_artifacts(
            metadata, metadata_path, manifest_path, session_root,
            session_id, mouse_id, metadata["protocol"], {
                "metadata": metadata_path,
                "planned_sequence": planned_sequence_path,
                "image_assignment": selected_images_path,
                "plan_summary": plan_summary_path,
                "event_log": event_log_path,
                "trial_summary": trial_summary_path,
                "lick_events": lick_events_path,
                "session_qc": qc_path,
                "camera_event_log": camera_event_log_path if use_camera else None,
                "video_manifest": (
                    session_root / "video" / "video_manifest.json"
                    if use_camera else None),
            }, {
                "task_completed": task_completed,
                "post_background_completed": post_background_completed,
                "interrupted": interrupted,
                "primary_error": primary_error,
                "camera_enabled": use_camera,
                "camera_state": latest_camera_state,
                "camera_cleanup_error": camera_cleanup_error,
            }, cleanup_errors, finalization_errors)
        camera_data_secured = completion_state["camera_data_secured"]
        session_completed = completion_state["session_completed"]

        if not session_completed:
            safe_report_telemetry_state(
                telemetry_state_reporter,
                "INTERRUPTED" if interrupted else "ERROR",
                total_trials=len(trials), total_blocks=n_blocks,
                eta_sec=None, water_accounting=water_accounting,
                force=True,
            )
        telemetry_publisher.close()

        if use_camera and camera_started:
            if camera_data_secured:
                print("VIDEO | camera data secured")
            elif (metadata["session_status"]
                  == "protocol_complete_camera_cleanup_failed"):
                print("VIDEO | camera cleanup failed; raw data retained when possible")
            else:
                print("VIDEO | camera data not yet secured; recovery pending")

        if session_completed:
            print("Session finished. Files are in: %s" % session_root)
        else:
            print("Session stopped early. Partial files are in: %s" % session_root)
        print("Session QC: %s" % qc_path)
        print("Trial summary: %s" % trial_summary_path)
        print(
            "For neural alignment, use the recorded photodiode/DAQ signal as "
            "ground truth; software timestamps are retained for diagnostics."
        )

    return final_session_exit(
        primary_error, interrupted, cleanup_errors, finalization_errors,
        camera_cleanup_error=camera_cleanup_error,
        camera_cleanup_error_message=camera_cleanup_error_message)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise
