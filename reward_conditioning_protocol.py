#!/usr/bin/env python3
"""Pure, hardware-free protocol logic for the exposure/reversal task."""

from __future__ import print_function

import fcntl
import hashlib
import json
import math
import os
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path


PROTOCOL_VERSION = "exposure_reward_partial_reversal_v1"
ASSIGNMENT_SCHEMA_VERSION = 3
PANEL_SCHEMA_VERSION = 2
DEFAULT_GLOBAL_PANEL_SEED = 777
DEFAULT_ASSIGNMENT_MASTER_SEED = 20260804
DEFAULT_SEQUENCE_MASTER_SEED = 20260805
GLOBAL_PANEL_FILENAME = "_exposure_reward_v1_14_image_panel.json"
ASSIGNMENT_LOCK_FILENAME = ".reward_conditioning_assignments.lock"

REWARDS_PER_TEN_PRESENTATIONS = 9
OMISSIONS_PER_TEN_PRESENTATIONS = 1
REWARD_PROBABILITY = 0.90
CONTINGENCY_PHASES = ("acquisition", "reversal")

HIGH_ROLES = ("high_R_to_R", "high_R_to_U", "high_U_to_R", "high_U_to_U")
MEDIUM_ROLES = ("medium_R_to_R", "medium_R_to_U", "medium_U_to_R", "medium_U_to_U")
LOW_ROLES = tuple("low_U_to_U_%02d" % index for index in range(1, 7))
ALL_ROLES = HIGH_ROLES + MEDIUM_ROLES + LOW_ROLES
ROLE_PRESENTATIONS_PER_BLOCK = dict(
    [(role, 8) for role in HIGH_ROLES]
    + [(role, 3) for role in MEDIUM_ROLES]
    + [(role, 1) for role in LOW_ROLES]
)
TRIALS_PER_BLOCK = sum(ROLE_PRESENTATIONS_PER_BLOCK.values())
assert TRIALS_PER_BLOCK == 50

REWARDED_ROLES_BY_PHASE = {
    "acquisition": ("high_R_to_R", "high_R_to_U", "medium_R_to_R", "medium_R_to_U"),
    "reversal": ("high_R_to_R", "high_U_to_R", "medium_R_to_R", "medium_U_to_R"),
}


def stable_seed(*parts):
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _role_metadata(role):
    if role in HIGH_ROLES:
        exposure_level, probability, presentations = "high", 0.16, 8
    elif role in MEDIUM_ROLES:
        exposure_level, probability, presentations = "medium", 0.06, 3
    elif role in LOW_ROLES:
        exposure_level, probability, presentations = "low", 0.02, 1
    else:
        raise ValueError("Unknown image role: %s" % role)
    trajectory = "U_to_U" if role in LOW_ROLES else role.split("_", 1)[1]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "image_role": role,
        "exposure_level": exposure_level,
        "image_category": exposure_level + "_exposure",
        "reward_trajectory": trajectory,
        "presentation_probability": probability,
        "presentations_per_block": presentations,
    }


def role_metadata(role):
    """Return permanent metadata for one image role."""
    return dict(_role_metadata(role))


def reward_state_for_role(image_role, contingency_phase):
    if contingency_phase not in CONTINGENCY_PHASES:
        raise ValueError("contingency_phase must be acquisition or reversal")
    trajectory = _role_metadata(image_role)["reward_trajectory"]
    return trajectory.split("_to_")[0 if contingency_phase == "acquisition" else 1]


def apply_contingency(assignment_rows, contingency_phase):
    if contingency_phase not in CONTINGENCY_PHASES:
        raise ValueError("contingency_phase must be acquisition or reversal")
    resolved = []
    for row in assignment_rows:
        current = dict(row)
        state = reward_state_for_role(row["image_role"], contingency_phase)
        current.update({
            "protocol_version": PROTOCOL_VERSION,
            "contingency_phase": contingency_phase,
            "current_reward_state": state,
            "reward_eligible": state == "R",
        })
        resolved.append(current)
    return resolved


def _parse_image_id(path):
    for part in reversed(Path(path).stem.split("_")):
        if part.isdigit():
            return int(part)
    return None


def assignment_path_for_mouse(assignment_dir, mouse_id):
    return Path(assignment_dir) / (str(mouse_id) + "_exposure_reward_v1_assignment.json")


def global_panel_path(assignment_dir):
    return Path(assignment_dir) / GLOBAL_PANEL_FILENAME


@contextmanager
def assignment_directory_lock(assignment_dir):
    assignment_dir = Path(assignment_dir)
    assignment_dir.mkdir(parents=True, exist_ok=True)
    with (assignment_dir / ASSIGNMENT_LOCK_FILENAME).open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield assignment_dir / ASSIGNMENT_LOCK_FILENAME
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(path.parent),
            prefix="." + path.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        temporary = None
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def _load_json(path, description):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError("Could not load valid JSON from %s %s: %s" % (description, path, error))


def _validate_global_panel_payload(payload, available_by_name, path):
    if not isinstance(payload, dict) or payload.get("schema_version") != PANEL_SCHEMA_VERSION:
        raise RuntimeError("Unsupported global panel schema in %s" % path)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(
            "Global panel protocol version mismatch in %s: %r"
            % (path, payload.get("protocol_version"))
        )
    filenames = payload.get("image_filenames", [])
    if len(filenames) != 14 or len(set(filenames)) != 14:
        raise RuntimeError("Global panel must contain 14 unique image filenames.")
    missing = [name for name in filenames if name not in available_by_name]
    if missing:
        raise RuntimeError("Global panel refers to missing image files: %s" % ", ".join(sorted(missing)))
    if isinstance(payload.get("panel_seed"), bool) or not isinstance(payload.get("panel_seed"), (int, str)):
        raise RuntimeError("Global panel has invalid panel_seed metadata.")
    try:
        int(payload["panel_seed"])
    except (KeyError, ValueError):
        raise RuntimeError("Global panel has invalid panel_seed metadata.")
    return list(filenames)


def _existing_assignment_paths(assignment_dir):
    return sorted(Path(assignment_dir).glob("*_exposure_reward_v1_assignment.json"))


def _create_or_load_global_panel_unlocked(available_image_files, assignment_dir, panel_seed=DEFAULT_GLOBAL_PANEL_SEED):
    available_image_files = sorted(Path(path) for path in available_image_files)
    available_by_name = {path.name: path for path in available_image_files}
    path = global_panel_path(assignment_dir)
    if path.exists():
        payload = _load_json(path, "global panel")
        filenames = _validate_global_panel_payload(payload, available_by_name, path)
        return [available_by_name[name] for name in filenames], path, False
    if _existing_assignment_paths(assignment_dir):
        raise RuntimeError("Refusing to create missing new global panel beside existing new assignments: %s" % path)
    if len(available_image_files) < 14:
        raise RuntimeError("Need at least 14 PNG files, but found %d." % len(available_image_files))
    selected = sorted(random.Random(int(panel_seed)).sample(available_image_files, 14))
    payload = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "panel_seed": int(panel_seed),
        "image_filenames": [path_item.name for path_item in selected],
    }
    _validate_global_panel_payload(payload, available_by_name, path)
    atomic_write_json(path, payload)
    return selected, path, True


def create_or_load_global_panel(available_image_files, assignment_dir, panel_seed=DEFAULT_GLOBAL_PANEL_SEED):
    with assignment_directory_lock(assignment_dir):
        return _create_or_load_global_panel_unlocked(available_image_files, assignment_dir, panel_seed)


def validate_assignment_rows(rows):
    if len(rows) != 14:
        raise RuntimeError("Assignment must contain exactly 14 images.")
    roles = [row.get("image_role") for row in rows]
    filenames = [row.get("image_filename") for row in rows]
    if sorted(roles) != sorted(ALL_ROLES):
        raise RuntimeError("Assignment roles are incomplete or duplicated.")
    if len(set(filenames)) != 14:
        raise RuntimeError("Assignment contains duplicated image filenames.")
    for row in rows:
        if row.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError("Assignment row belongs to another protocol version.")
        expected = _role_metadata(row["image_role"])
        for field in ("exposure_level", "reward_trajectory", "presentation_probability", "presentations_per_block"):
            if row.get(field) != expected[field]:
                raise RuntimeError("Assignment metadata does not match role %s." % row["image_role"])


def _resolve_assignment_images(payload, available_image_files):
    available_by_name = {Path(path).name: Path(path) for path in available_image_files}
    rows, missing = [], []
    for saved_row in payload.get("images", []):
        if not isinstance(saved_row, dict) or "image_filename" not in saved_row:
            raise RuntimeError("Saved assignment contains an invalid image row.")
        if saved_row["image_filename"] not in available_by_name:
            missing.append(saved_row["image_filename"])
            continue
        row = dict(saved_row)
        row["image_path"] = str(available_by_name[row["image_filename"]])
        rows.append(row)
    if missing:
        raise RuntimeError("The saved assignment refers to missing image files: %s" % ", ".join(sorted(missing)))
    return rows


def _validate_assignment_payload(payload, expected_mouse_id, available_image_files, assignment_path, panel_filenames, panel_path, panel_seed):
    if not isinstance(payload, dict) or payload.get("schema_version") != ASSIGNMENT_SCHEMA_VERSION:
        raise RuntimeError("Unsupported assignment schema in %s" % assignment_path)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Assignment protocol version mismatch in %s" % assignment_path)
    if payload.get("mouse_id") != expected_mouse_id:
        raise RuntimeError("Assignment mouse mismatch: expected %s, found %s" % (expected_mouse_id, payload.get("mouse_id")))
    rows = _resolve_assignment_images(payload, available_image_files)
    validate_assignment_rows(rows)
    if {row["image_filename"] for row in rows} != set(panel_filenames):
        raise RuntimeError("Assignment filenames do not match the authoritative global panel.")
    saved_panel_path = payload.get("global_panel_path")
    if not isinstance(saved_panel_path, str) or Path(saved_panel_path).name != Path(panel_path).name:
        raise RuntimeError("Assignment refers to an inconsistent global panel path.")
    try:
        saved_seed = int(payload["global_panel_seed"])
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("Assignment has invalid global panel seed metadata.")
    if saved_seed != int(panel_seed):
        raise RuntimeError("Assignment global panel seed does not match the authoritative global panel seed.")
    return rows


def create_or_load_assignment(mouse_id, available_image_files, assignment_dir, master_seed=DEFAULT_ASSIGNMENT_MASTER_SEED, panel_seed=DEFAULT_GLOBAL_PANEL_SEED, force_new=False):
    mouse_id = str(mouse_id)
    available_image_files = sorted(Path(path) for path in available_image_files)
    if len(available_image_files) < 14:
        raise RuntimeError("Need at least 14 PNG files, but found %d." % len(available_image_files))
    with assignment_directory_lock(assignment_dir):
        panel_files, panel_path, panel_created = _create_or_load_global_panel_unlocked(available_image_files, assignment_dir, panel_seed)
        panel_payload = _load_json(panel_path, "global panel")
        available_by_name = {path.name: path for path in available_image_files}
        panel_filenames = _validate_global_panel_payload(panel_payload, available_by_name, panel_path)
        authoritative_seed = int(panel_payload["panel_seed"])
        assignment_path = assignment_path_for_mouse(assignment_dir, mouse_id)
        if assignment_path.exists() and not force_new:
            payload = _load_json(assignment_path, "mouse assignment")
            rows = _validate_assignment_payload(payload, mouse_id, available_image_files, assignment_path, panel_filenames, panel_path, authoritative_seed)
            return rows, assignment_path, False, payload.get("resolved_assignment_seed")
        resolved_seed = stable_seed("exposure-reward-role-assignment", master_seed, mouse_id)
        shuffled = list(panel_files)
        random.Random(resolved_seed).shuffle(shuffled)
        rows = []
        for role, path in zip(ALL_ROLES, shuffled):
            row = _role_metadata(role)
            row.update({"image_filename": path.name, "image_path": str(path), "image_id": _parse_image_id(path)})
            rows.append(row)
        validate_assignment_rows(rows)
        payload = {
            "schema_version": ASSIGNMENT_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "mouse_id": mouse_id,
            "assignment_master_seed": int(master_seed),
            "resolved_assignment_seed": int(resolved_seed),
            "global_panel_seed": authoritative_seed,
            "global_panel_path": str(panel_path),
            "global_panel_created_with_this_assignment": bool(panel_created),
            "images": rows,
        }
        atomic_write_json(assignment_path, payload)
        return rows, assignment_path, True, resolved_seed


def _constrained_shuffle(rows, rng, previous_filename=None, max_attempts=1000):
    """Shuffle a multiset without equal neighbors, including block boundaries."""
    rows = [dict(row) for row in rows]
    for _ in range(max_attempts):
        buckets = {}
        for row in rows:
            buckets.setdefault(row["image_filename"], []).append(row)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        sequence, previous = [], previous_filename
        while buckets:
            remaining = sum(len(bucket) for bucket in buckets.values())
            candidates = []
            for filename, bucket in buckets.items():
                if filename == previous:
                    continue
                after = remaining - 1
                largest = max(len(other) - (filename == other_name) for other_name, other in buckets.items())
                if largest <= int(math.ceil(after / 2.0)):
                    candidates.append(filename)
            if not candidates:
                break
            chosen = rng.choice([name for name in candidates for _ in buckets[name]])
            sequence.append(buckets[chosen].pop())
            if not buckets[chosen]:
                del buckets[chosen]
            previous = chosen
        if len(sequence) == len(rows):
            return sequence
    raise RuntimeError("Could not construct a block without adjacent repeated images.")


def _stratified_omission_indices(n_presentations, n_omissions, rng):
    if n_omissions <= 0:
        return set()
    if n_omissions >= n_presentations:
        return set(range(n_presentations))
    for _ in range(10000):
        chosen = []
        for stratum in range(n_omissions):
            start = int(math.floor(stratum * n_presentations / float(n_omissions)))
            stop = max(start + 1, int(math.floor((stratum + 1) * n_presentations / float(n_omissions))))
            chosen.append(rng.randrange(start, min(n_presentations, stop)))
        chosen = sorted(set(chosen))
        if len(chosen) == n_omissions and all(b - a > 1 for a, b in zip(chosen, chosen[1:])):
            return set(chosen)
    return set(rng.sample(range(n_presentations), n_omissions))


def _assign_exact_rewards(trials, contingency_phase, rng):
    rewarded_roles = set(REWARDED_ROLES_BY_PHASE[contingency_phase])
    for role in ALL_ROLES:
        matching = [index for index, trial in enumerate(trials) if trial["image_role"] == role]
        if role not in rewarded_roles:
            for index in matching:
                trials[index].update({"reward_scheduled": False, "reward_omission_scheduled": False, "rewarded_cue_presentation_ordinal": ""})
            continue
        if len(matching) % 10:
            raise RuntimeError("%s has a presentation count not divisible by 10." % role)
        omission_indices = _stratified_omission_indices(len(matching), len(matching) // 10, rng)
        for ordinal, index in enumerate(matching):
            omitted = ordinal in omission_indices
            trials[index].update({"reward_scheduled": not omitted, "reward_omission_scheduled": omitted, "rewarded_cue_presentation_ordinal": ordinal + 1})


def make_trial_plan(assignment_rows, n_blocks, iti_min_sec=3.0, iti_max_sec=4.5, sequence_seed=None, mouse_id="mouse", sequence_master_seed=DEFAULT_SEQUENCE_MASTER_SEED, stim_duration_sec=1.5, reward_delay_sec=1.0, suction_delay_sec=3.5, contingency_phase="acquisition"):
    validate_assignment_rows(assignment_rows)
    if contingency_phase not in CONTINGENCY_PHASES:
        raise ValueError("contingency_phase must be acquisition or reversal")
    if int(n_blocks) < 1:
        raise ValueError("n_blocks must be at least 1")
    if int(n_blocks) % 10:
        raise ValueError("n_blocks must be a multiple of 10 for exact per-role 90% scheduling")
    if float(iti_min_sec) <= 0 or float(iti_max_sec) < float(iti_min_sec):
        raise ValueError("Require 0 < iti_min_sec <= iti_max_sec")
    if not 0.0 < float(reward_delay_sec) < float(stim_duration_sec):
        raise ValueError("reward_delay_sec must be inside the stimulus interval")
    if not math.isfinite(float(suction_delay_sec)) or float(suction_delay_sec) < float(stim_duration_sec):
        raise ValueError("suction_delay_sec must be finite and at least stim_duration_sec")
    sequence_seed = sequence_seed if sequence_seed is not None else stable_seed("exposure-reward-sequence", sequence_master_seed, mouse_id, random.SystemRandom().getrandbits(64))
    rng = random.Random(int(sequence_seed))
    resolved_rows = apply_contingency(assignment_rows, contingency_phase)
    by_role = {row["image_role"]: row for row in resolved_rows}
    trials, previous_filename = [], None
    for block_index in range(int(n_blocks)):
        block_rows = []
        for role in ALL_ROLES:
            block_rows.extend([dict(by_role[role]) for _ in range(ROLE_PRESENTATIONS_PER_BLOCK[role])])
        block_rows = _constrained_shuffle(block_rows, rng, previous_filename)
        previous_filename = block_rows[-1]["image_filename"]
        for within, image_row in enumerate(block_rows):
            trial = dict(image_row)
            trial.update({
                "trial_index": len(trials), "trial_number": len(trials) + 1,
                "block_index": block_index, "block_number": block_index + 1,
                "within_block_index": within, "within_block_number": within + 1,
                "planned_stim_duration_sec": float(stim_duration_sec),
                "planned_reward_delay_sec": float(reward_delay_sec),
                "planned_post_reward_stim_sec": float(stim_duration_sec) - float(reward_delay_sec),
                "planned_iti_duration_sec": round(rng.uniform(float(iti_min_sec), float(iti_max_sec)), 6),
                "suction_scheduled": bool(image_row["reward_eligible"]),
                "planned_suction_delay_sec": float(suction_delay_sec),
                "reward_scheduled": False, "reward_omission_scheduled": False,
            })
            trials.append(trial)
    _assign_exact_rewards(trials, contingency_phase, random.Random(stable_seed("reward-schedule", sequence_seed, contingency_phase)))
    validate_trial_plan(trials, int(n_blocks), contingency_phase)
    return trials, int(sequence_seed)


def validate_trial_plan(trials, n_blocks, contingency_phase=None):
    contingency_phase = contingency_phase or (trials[0].get("contingency_phase") if trials else "acquisition")
    if contingency_phase not in CONTINGENCY_PHASES:
        raise RuntimeError("Invalid contingency phase in trial plan.")
    expected_total = int(n_blocks) * TRIALS_PER_BLOCK
    if len(trials) != expected_total:
        raise RuntimeError("Expected %d trials, but generated %d." % (expected_total, len(trials)))
    if any(trial.get("contingency_phase") != contingency_phase for trial in trials):
        raise RuntimeError("Trial plan contains mixed contingency phases.")
    for previous, current in zip(trials, trials[1:]):
        if previous["image_filename"] == current["image_filename"]:
            raise RuntimeError("Adjacent repeated image at trial index %d." % current["trial_index"])
    rewarded_roles = set(REWARDED_ROLES_BY_PHASE[contingency_phase])
    for block_index in range(int(n_blocks)):
        block = [trial for trial in trials if trial["block_index"] == block_index]
        if len(block) != TRIALS_PER_BLOCK:
            raise RuntimeError("Block %d does not contain 50 trials." % (block_index + 1))
        for role, expected in ROLE_PRESENTATIONS_PER_BLOCK.items():
            observed = sum(trial["image_role"] == role for trial in block)
            if observed != expected:
                raise RuntimeError("Block %d role %s: expected %d, observed %d." % (block_index + 1, role, expected, observed))
    for trial in trials:
        eligible = trial["image_role"] in rewarded_roles
        if bool(trial["reward_eligible"]) != eligible or bool(trial["suction_scheduled"]) != eligible:
            raise RuntimeError("Phase-resolved reward metadata is inconsistent.")
        if trial["reward_scheduled"] and trial["reward_omission_scheduled"]:
            raise RuntimeError("Reward and omission are both scheduled on one trial.")
        if eligible and bool(trial["reward_scheduled"]) == bool(trial["reward_omission_scheduled"]):
            raise RuntimeError("Current R+ trial must schedule exactly one reward outcome.")
        if not eligible and (trial["reward_scheduled"] or trial["reward_omission_scheduled"]):
            raise RuntimeError("Reward or omission scheduled on a current R- trial.")
    for role in rewarded_roles:
        role_trials = [trial for trial in trials if trial["image_role"] == role]
        if sum(trial["reward_scheduled"] for trial in role_trials) != len(role_trials) * 9 // 10 or sum(trial["reward_omission_scheduled"] for trial in role_trials) != len(role_trials) // 10:
            raise RuntimeError("%s has an invalid exact reward schedule." % role)
    if sum(trial["reward_scheduled"] for trial in trials) != int(n_blocks) * 198 // 10:
        raise RuntimeError("Trial plan has an invalid total reward schedule.")


def summarize_trial_plan(trials):
    rows = []
    for role in ALL_ROLES:
        role_trials = [trial for trial in trials if trial["image_role"] == role]
        if not role_trials:
            continue
        rows.append({
            "image_role": role,
            "image_filename": role_trials[0]["image_filename"],
            "exposure_level": role_trials[0]["exposure_level"],
            "reward_trajectory": role_trials[0]["reward_trajectory"],
            "n_presentations": len(role_trials),
            "n_rewards": sum(trial["reward_scheduled"] for trial in role_trials),
            "n_omissions": sum(trial["reward_omission_scheduled"] for trial in role_trials),
            "realized_reward_probability": sum(trial["reward_scheduled"] for trial in role_trials) / float(len(role_trials)) if role_trials[0]["reward_eligible"] else 0.0,
        })
    return rows


if __name__ == "__main__":
    files = [Path("natural_image_%04d.png" % index) for index in range(100)]
    directory = Path(tempfile.mkdtemp(prefix="reward_conditioning_protocol_"))
    rows, _, _, _ = create_or_load_assignment("TEST_MOUSE", files, directory)
    trials, _ = make_trial_plan(rows, 10, sequence_seed=12345)
    assert len(trials) == 500
    assert sum(trial["reward_scheduled"] for trial in trials) == 198
    print("protocol smoke test: 14 roles, 500 trials, 198 rewards")
