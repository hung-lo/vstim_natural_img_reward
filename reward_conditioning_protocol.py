#!/usr/bin/env python3
"""Pure protocol logic for natural-image reward conditioning.

This module has no Raspberry Pi, RPG, camera, or GPIO dependency.  It owns:

* one persistent 14-image panel shared across the cohort;
* persistent image-to-role assignment for each mouse;
* exact 50-trial probability blocks;
* constrained pseudorandomization with no adjacent image repeats;
* exact 90% reward / 10% omission scheduling independent of licking; and
* validation of the complete planned session before hardware starts.

The intended protocol has 14 images:

* 10 low-probability images: 1 presentation each per 50-trial block (2% each)
* 2 high-probability unrewarded images: 10 each per block (20% each)
* 2 high-probability rewarded images: 10 each per block (20% each)
"""

from __future__ import print_function

import hashlib
import json
import math
import random
from pathlib import Path


ASSIGNMENT_SCHEMA_VERSION = 2
PANEL_SCHEMA_VERSION = 1
DEFAULT_GLOBAL_PANEL_SEED = 777
DEFAULT_ASSIGNMENT_MASTER_SEED = 20260804
DEFAULT_SEQUENCE_MASTER_SEED = 20260805
GLOBAL_PANEL_FILENAME = "_global_reward_conditioning_14_image_panel.json"

# Protocol version 1 deliberately fixes this schedule. Do not expose it as
# an operator prompt, command-line option, or hardware-configuration field.
REWARDS_PER_TEN_PRESENTATIONS = 9
OMISSIONS_PER_TEN_PRESENTATIONS = 1
REWARD_PROBABILITY = REWARDS_PER_TEN_PRESENTATIONS / 10.0

REWARDED_HIGH_ROLES = ("rewarded_high_1", "rewarded_high_2")
UNREWARDED_HIGH_ROLES = ("unrewarded_high_1", "unrewarded_high_2")
LOW_ROLES = tuple("low_%02d" % index for index in range(1, 11))
ALL_ROLES = REWARDED_HIGH_ROLES + UNREWARDED_HIGH_ROLES + LOW_ROLES

ROLE_PRESENTATIONS_PER_BLOCK = {}
for _role in REWARDED_HIGH_ROLES + UNREWARDED_HIGH_ROLES:
    ROLE_PRESENTATIONS_PER_BLOCK[_role] = 10
for _role in LOW_ROLES:
    ROLE_PRESENTATIONS_PER_BLOCK[_role] = 1

TRIALS_PER_BLOCK = sum(ROLE_PRESENTATIONS_PER_BLOCK.values())
assert TRIALS_PER_BLOCK == 50


def stable_seed(*parts):
    """Return a stable integer seed from arbitrary values."""
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _role_metadata(role):
    if role in REWARDED_HIGH_ROLES:
        return {
            "image_role": role,
            "image_category": "high_probability_rewarded",
            "presentation_probability": 0.20,
            "reward_eligible": True,
        }
    if role in UNREWARDED_HIGH_ROLES:
        return {
            "image_role": role,
            "image_category": "high_probability_unrewarded",
            "presentation_probability": 0.20,
            "reward_eligible": False,
        }
    if role in LOW_ROLES:
        return {
            "image_role": role,
            "image_category": "low_probability_unrewarded",
            "presentation_probability": 0.02,
            "reward_eligible": False,
        }
    raise ValueError("Unknown image role: %s" % role)


def _parse_image_id(path):
    path = Path(path)
    for part in reversed(path.stem.split("_")):
        if part.isdigit():
            return int(part)
    return None


def assignment_path_for_mouse(assignment_dir, mouse_id):
    return Path(assignment_dir) / (str(mouse_id) + "_reward_conditioning_assignment.json")


def global_panel_path(assignment_dir):
    return Path(assignment_dir) / GLOBAL_PANEL_FILENAME


def create_or_load_global_panel(
    available_image_files,
    assignment_dir,
    panel_seed=DEFAULT_GLOBAL_PANEL_SEED,
):
    """Create or load one shared 14-image panel for the entire cohort.

    Using the same image identities across mice improves cross-animal comparison.
    Image-to-role mappings are still independently counterbalanced per mouse.
    """
    available_image_files = sorted(Path(path) for path in available_image_files)
    available_by_name = {path.name: path for path in available_image_files}
    assignment_dir = Path(assignment_dir)
    assignment_dir.mkdir(parents=True, exist_ok=True)
    path = global_panel_path(assignment_dir)

    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != PANEL_SCHEMA_VERSION:
            raise RuntimeError(
                "Unsupported global panel schema in %s: %r"
                % (path, payload.get("schema_version"))
            )
        filenames = list(payload.get("image_filenames", []))
        if len(filenames) != len(ALL_ROLES) or len(set(filenames)) != len(filenames):
            raise RuntimeError("Global panel must contain 14 unique image filenames.")
        missing = [name for name in filenames if name not in available_by_name]
        if missing:
            raise RuntimeError(
                "Global panel refers to missing image files: %s"
                % ", ".join(sorted(missing))
            )
        return [available_by_name[name] for name in filenames], path, False

    if len(available_image_files) < len(ALL_ROLES):
        raise RuntimeError(
            "Need at least %d PNG files, but found %d."
            % (len(ALL_ROLES), len(available_image_files))
        )
    rng = random.Random(int(panel_seed))
    selected = sorted(rng.sample(available_image_files, len(ALL_ROLES)))
    payload = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "panel_seed": int(panel_seed),
        "image_filenames": [path_item.name for path_item in selected],
        "notes": (
            "Shared 14-image cohort panel. Do not replace after data collection "
            "begins. Per-mouse files assign these images to protocol roles."
        ),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return selected, path, True


def _resolve_assignment_images(payload, available_image_files):
    """Resolve saved filenames against the currently available image directory."""
    available_by_name = {Path(path).name: Path(path) for path in available_image_files}
    resolved_rows = []
    missing = []
    for saved_row in payload.get("images", []):
        filename = saved_row["image_filename"]
        if filename not in available_by_name:
            missing.append(filename)
            continue
        row = dict(saved_row)
        row["image_path"] = str(available_by_name[filename])
        resolved_rows.append(row)
    if missing:
        raise RuntimeError(
            "The saved assignment refers to missing image files: %s" % ", ".join(sorted(missing))
        )
    return resolved_rows


def create_or_load_assignment(
    mouse_id,
    available_image_files,
    assignment_dir,
    master_seed=DEFAULT_ASSIGNMENT_MASTER_SEED,
    panel_seed=DEFAULT_GLOBAL_PANEL_SEED,
    force_new=False,
):
    """Create or load a fixed 14-image assignment for one mouse.

    Existing assignments are always reused unless ``force_new`` is explicitly
    requested.  The saved filenames, not newly sampled images, are treated as
    authoritative for longitudinal experiments.
    """
    mouse_id = str(mouse_id)
    available_image_files = sorted(Path(path) for path in available_image_files)
    if len(available_image_files) < len(ALL_ROLES):
        raise RuntimeError(
            "Need at least %d PNG files, but found %d."
            % (len(ALL_ROLES), len(available_image_files))
        )

    assignment_dir = Path(assignment_dir)
    assignment_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = assignment_path_for_mouse(assignment_dir, mouse_id)

    if assignment_path.exists() and not force_new:
        payload = json.loads(assignment_path.read_text())
        if payload.get("schema_version") != ASSIGNMENT_SCHEMA_VERSION:
            raise RuntimeError(
                "Unsupported assignment schema in %s: %r"
                % (assignment_path, payload.get("schema_version"))
            )
        if payload.get("mouse_id") != mouse_id:
            raise RuntimeError(
                "Assignment mouse mismatch: expected %s, found %s"
                % (mouse_id, payload.get("mouse_id"))
            )
        rows = _resolve_assignment_images(payload, available_image_files)
        validate_assignment_rows(rows)
        return rows, assignment_path, False, payload.get("resolved_assignment_seed")

    panel_files, panel_path, panel_created = create_or_load_global_panel(
        available_image_files,
        assignment_dir,
        panel_seed=panel_seed,
    )
    resolved_seed = stable_seed("reward-conditioning-role-assignment", master_seed, mouse_id)
    rng = random.Random(resolved_seed)
    role_ordered_files = list(panel_files)
    rng.shuffle(role_ordered_files)

    rows = []
    for role, path in zip(ALL_ROLES, role_ordered_files):
        row = _role_metadata(role)
        row.update(
            {
                "image_filename": path.name,
                "image_path": str(path),
                "image_id": _parse_image_id(path),
            }
        )
        rows.append(row)

    validate_assignment_rows(rows)
    payload = {
        "schema_version": ASSIGNMENT_SCHEMA_VERSION,
        "mouse_id": mouse_id,
        "assignment_master_seed": int(master_seed),
        "resolved_assignment_seed": int(resolved_seed),
        "global_panel_seed": int(panel_seed),
        "global_panel_path": str(panel_path),
        "global_panel_created_with_this_assignment": bool(panel_created),
        "images": rows,
        "notes": (
            "This file fixes image identities and roles for longitudinal use. "
            "Do not delete or regenerate it after training begins."
        ),
    }
    assignment_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return rows, assignment_path, True, resolved_seed


def validate_assignment_rows(rows):
    if len(rows) != len(ALL_ROLES):
        raise RuntimeError("Assignment must contain exactly 14 images.")
    roles = [row["image_role"] for row in rows]
    filenames = [row["image_filename"] for row in rows]
    if sorted(roles) != sorted(ALL_ROLES):
        raise RuntimeError("Assignment roles are incomplete or duplicated.")
    if len(set(filenames)) != len(filenames):
        raise RuntimeError("Assignment contains duplicated image filenames.")


def _constrained_shuffle(rows, rng, previous_filename=None, max_attempts=1000):
    """Randomized greedy shuffle with no adjacent identical image.

    A pure rejection shuffle is inefficient here because four images each occupy
    20% of a block.  This builder chooses among candidates that preserve the
    standard no-adjacent feasibility condition, then retries only if a rare
    greedy dead end is reached.
    """
    rows = [dict(row) for row in rows]
    for _ in range(max_attempts):
        buckets = {}
        for row in rows:
            buckets.setdefault(row["image_filename"], []).append(dict(row))
        for bucket in buckets.values():
            rng.shuffle(bucket)

        sequence = []
        previous = previous_filename
        while buckets:
            total_remaining = sum(len(bucket) for bucket in buckets.values())
            candidates = []
            for filename, bucket in buckets.items():
                if filename == previous:
                    continue
                # Test the remaining multiset after choosing this filename.
                remaining_after = total_remaining - 1
                max_after = 0
                for other_filename, other_bucket in buckets.items():
                    count_after = len(other_bucket) - (1 if other_filename == filename else 0)
                    max_after = max(max_after, count_after)
                if max_after <= int(math.ceil(remaining_after / 2.0)):
                    candidates.append(filename)

            if not candidates:
                break

            # Weighted choice retains randomness while favoring abundant images.
            weighted = []
            for filename in candidates:
                weighted.extend([filename] * len(buckets[filename]))
            chosen = rng.choice(weighted)
            sequence.append(buckets[chosen].pop())
            if not buckets[chosen]:
                del buckets[chosen]
            previous = chosen

        if len(sequence) == len(rows):
            return sequence

    raise RuntimeError("Could not construct a block without adjacent repeated images.")


def _stratified_omission_indices(n_presentations, n_omissions, rng):
    """Place an exact number of omissions approximately evenly through a cue's trials."""
    if n_omissions <= 0:
        return set()
    if n_omissions >= n_presentations:
        return set(range(n_presentations))

    # Each omission is sampled from a separate temporal stratum.  Retry if two
    # choices become adjacent; high reward probabilities make this easy.
    for _ in range(10000):
        chosen = []
        for stratum in range(n_omissions):
            start = int(math.floor(stratum * n_presentations / float(n_omissions)))
            stop_exclusive = int(
                math.floor((stratum + 1) * n_presentations / float(n_omissions))
            )
            stop_exclusive = max(start + 1, stop_exclusive)
            chosen.append(rng.randrange(start, min(n_presentations, stop_exclusive)))
        chosen = sorted(set(chosen))
        if len(chosen) != n_omissions:
            continue
        if all(chosen[index] - chosen[index - 1] > 1 for index in range(1, len(chosen))):
            return set(chosen)

    # Exact count is more important than the no-adjacent-omission preference.
    return set(rng.sample(range(n_presentations), n_omissions))


def _assign_exact_rewards(trials, rng):
    for role in REWARDED_HIGH_ROLES:
        matching_indices = [
            index for index, trial in enumerate(trials) if trial["image_role"] == role
        ]
        n_presentations = len(matching_indices)
        if n_presentations % 10 != 0:
            raise RuntimeError(
                "%s has %d presentations; fixed 90%% scheduling requires a "
                "multiple of 10." % (role, n_presentations)
            )
        groups_of_ten = n_presentations // 10
        n_rewards = groups_of_ten * REWARDS_PER_TEN_PRESENTATIONS
        n_omissions = groups_of_ten * OMISSIONS_PER_TEN_PRESENTATIONS
        omission_ordinals = _stratified_omission_indices(
            n_presentations, n_omissions, rng
        )
        for ordinal, trial_index in enumerate(matching_indices):
            scheduled = ordinal not in omission_ordinals
            trials[trial_index]["reward_scheduled"] = bool(scheduled)
            trials[trial_index]["reward_omission_scheduled"] = bool(not scheduled)
            trials[trial_index]["rewarded_cue_presentation_ordinal"] = ordinal + 1

    for trial in trials:
        if not trial["reward_eligible"]:
            trial["reward_scheduled"] = False
            trial["reward_omission_scheduled"] = False
            trial["rewarded_cue_presentation_ordinal"] = ""


def make_trial_plan(
    assignment_rows,
    n_blocks,
    iti_min_sec=3.0,
    iti_max_sec=4.5,
    sequence_seed=None,
    mouse_id="mouse",
    sequence_master_seed=DEFAULT_SEQUENCE_MASTER_SEED,
    stim_duration_sec=1.5,
    reward_delay_sec=1.0,
    suction_delay_sec=3.5,
):
    """Build and validate the complete planned session before hardware starts."""
    validate_assignment_rows(assignment_rows)
    if int(n_blocks) < 1:
        raise ValueError("n_blocks must be at least 1")
    if float(iti_min_sec) <= 0 or float(iti_max_sec) < float(iti_min_sec):
        raise ValueError("Require 0 < iti_min_sec <= iti_max_sec")
    if not 0.0 < float(reward_delay_sec) < float(stim_duration_sec):
        raise ValueError("reward_delay_sec must be inside the stimulus interval")
    if not math.isfinite(float(suction_delay_sec)) or float(suction_delay_sec) < float(stim_duration_sec):
        raise ValueError("suction_delay_sec must be finite and at least stim_duration_sec")

    if sequence_seed is None:
        sequence_seed = stable_seed(
            "reward-conditioning-sequence",
            sequence_master_seed,
            mouse_id,
            random.SystemRandom().getrandbits(64),
        )
    rng = random.Random(int(sequence_seed))
    assignment_by_role = {row["image_role"]: dict(row) for row in assignment_rows}

    trials = []
    previous_filename = None
    for block_index in range(int(n_blocks)):
        block_rows = []
        for role in ALL_ROLES:
            for _ in range(ROLE_PRESENTATIONS_PER_BLOCK[role]):
                block_rows.append(dict(assignment_by_role[role]))
        block_rows = _constrained_shuffle(
            block_rows, rng, previous_filename=previous_filename
        )
        previous_filename = block_rows[-1]["image_filename"]

        for within_block_index, image_row in enumerate(block_rows):
            trial = dict(image_row)
            trial.update(
                {
                    "trial_index": len(trials),
                    "trial_number": len(trials) + 1,
                    "block_index": block_index,
                    "block_number": block_index + 1,
                    "within_block_index": within_block_index,
                    "within_block_number": within_block_index + 1,
                    "planned_stim_duration_sec": float(stim_duration_sec),
                    "planned_reward_delay_sec": float(reward_delay_sec),
                    "planned_post_reward_stim_sec": float(stim_duration_sec)
                    - float(reward_delay_sec),
                    "planned_iti_duration_sec": round(
                        rng.uniform(float(iti_min_sec), float(iti_max_sec)), 6
                    ),
                    "suction_scheduled": bool(image_row["reward_eligible"]),
                    "planned_suction_delay_sec": float(suction_delay_sec),
                    "reward_scheduled": False,
                    "reward_omission_scheduled": False,
                }
            )
            trials.append(trial)

    reward_rng = random.Random(stable_seed("reward-schedule", sequence_seed))
    _assign_exact_rewards(trials, reward_rng)
    validate_trial_plan(trials, n_blocks=int(n_blocks))
    return trials, int(sequence_seed)


def validate_trial_plan(trials, n_blocks):
    expected_total = int(n_blocks) * TRIALS_PER_BLOCK
    if len(trials) != expected_total:
        raise RuntimeError(
            "Expected %d trials, but generated %d." % (expected_total, len(trials))
        )

    for index in range(1, len(trials)):
        if trials[index]["image_filename"] == trials[index - 1]["image_filename"]:
            raise RuntimeError("Adjacent repeated image at trial index %d." % index)

    for block_index in range(int(n_blocks)):
        block = [trial for trial in trials if trial["block_index"] == block_index]
        if len(block) != TRIALS_PER_BLOCK:
            raise RuntimeError("Block %d does not contain 50 trials." % (block_index + 1))
        for role, expected_count in ROLE_PRESENTATIONS_PER_BLOCK.items():
            observed = sum(1 for trial in block if trial["image_role"] == role)
            if observed != expected_count:
                raise RuntimeError(
                    "Block %d role %s: expected %d, observed %d."
                    % (block_index + 1, role, expected_count, observed)
                )

    for trial in trials:
        if not trial["reward_eligible"] and trial["reward_scheduled"]:
            raise RuntimeError("Reward scheduled on an ineligible trial.")
        if bool(trial.get("suction_scheduled")) != bool(trial["reward_eligible"]):
            raise RuntimeError("Suction must be scheduled exactly for reward-associated trials.")

    for role in REWARDED_HIGH_ROLES:
        role_trials = [trial for trial in trials if trial["image_role"] == role]
        if len(role_trials) % 10 != 0:
            raise RuntimeError(
                "%s has %d presentations; expected a multiple of 10."
                % (role, len(role_trials))
            )
        groups_of_ten = len(role_trials) // 10
        expected_rewards = groups_of_ten * REWARDS_PER_TEN_PRESENTATIONS
        expected_omissions = (
            groups_of_ten * OMISSIONS_PER_TEN_PRESENTATIONS
        )
        observed_rewards = sum(1 for trial in role_trials if trial["reward_scheduled"])
        observed_omissions = sum(
            1 for trial in role_trials if trial["reward_omission_scheduled"]
        )
        if observed_rewards != expected_rewards:
            raise RuntimeError(
                "%s: expected %d rewards, observed %d."
                % (role, expected_rewards, observed_rewards)
            )
        if observed_omissions != expected_omissions:
            raise RuntimeError(
                "%s: expected %d omissions, observed %d."
                % (role, expected_omissions, observed_omissions)
            )
        if any(
            bool(trial["reward_scheduled"])
            == bool(trial["reward_omission_scheduled"])
            for trial in role_trials
        ):
            raise RuntimeError(
                "%s contains a rewarded-cue trial without exactly one of "
                "reward_scheduled or reward_omission_scheduled." % role
            )


def summarize_trial_plan(trials):
    rows = []
    for role in ALL_ROLES:
        role_trials = [trial for trial in trials if trial["image_role"] == role]
        if not role_trials:
            continue
        rewards = sum(1 for trial in role_trials if trial["reward_scheduled"])
        omissions = sum(1 for trial in role_trials if trial["reward_omission_scheduled"])
        rows.append(
            {
                "image_role": role,
                "image_filename": role_trials[0]["image_filename"],
                "n_presentations": len(role_trials),
                "n_rewards": rewards,
                "n_omissions": omissions,
                "realized_reward_probability": (
                    rewards / float(len(role_trials)) if role_trials[0]["reward_eligible"] else 0.0
                ),
            }
        )
    return rows


if __name__ == "__main__":
    # Hardware-free smoke test.
    fake_files = [Path("natural_image_%04d.png" % index) for index in range(100)]
    temp_dir = Path("/tmp/reward_conditioning_protocol_test")
    rows, path, created, assignment_seed = create_or_load_assignment(
        "TEST_MOUSE", fake_files, temp_dir, force_new=True
    )
    trials, sequence_seed = make_trial_plan(
        rows,
        n_blocks=10,
        iti_min_sec=2.5,
        iti_max_sec=4.5,
        mouse_id="TEST_MOUSE",
        sequence_seed=12345,
    )
    print("Assignment:", path, "created=", created, "seed=", assignment_seed)
    print("Sequence seed:", sequence_seed)
    print("Trials:", len(trials))
    for summary in summarize_trial_plan(trials):
        print(summary)
