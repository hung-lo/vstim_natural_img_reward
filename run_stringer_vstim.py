#!/usr/bin/env python3
"""
run_stringer_vstim.py

Headless visual stimulus runner for the behavior Raspberry Pi.
This version uses the lab's rpg framebuffer path rather than pygame.
"""

import sys

if __name__ == "__main__":
    print(
        "This repository's supported experiment entrypoint is:\n\n"
        "  python3 run_stringer_reward_conditioning.py\n\n"
        "For the natural-image-only protocol use:\n"
        "  https://github.com/hung-lo/vstim_natural",
        file=sys.stderr,
    )
    raise SystemExit(2)

import csv
import json
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_DIR_CANDIDATES = [
    PROJECT_ROOT / "stringer_natimg2800_center_crop_png",
    Path.home() / "vstim_natural" / "stringer_natimg2800_center_crop_png",
    Path.home() / "stringer_natimg2800_center_crop_png",
]
OUTPUT_ROOT = Path("/mnt/hd")
SESSION_NAME_SUFFIX = "vstim_natural"

SCREEN_RESOLUTION = (1024, 600)
SCREEN_BACKGROUND_GRAY = 127
SCREEN_COLORMODE = 16
REFRESH_RATE_HZ = 60

N_IMAGES_TO_USE = 5
N_REPEATS = 2
IMAGE_SUBSET_SEED = 777
TRIAL_ORDER_SEED = None
AVOID_ADJACENT_REPEATS = True

STIM_DURATION_SEC = 0.5
ITI_DURATION_SEC = 0.75
INITIAL_GRAY_SEC = 3.0
FINAL_GRAY_SEC = 3.0
POSTSTIM_GRAY_PLANNED_SEC = FINAL_GRAY_SEC
POSTSTIM_BLACK_LEVEL = 0
POSTSTIM_TRANSITION_GRAY_SEC = POSTSTIM_GRAY_PLANNED_SEC

ENABLE_PHOTODIODE_PATCH = True
PHOTODIODE_SIZE_PX = 120
PHOTODIODE_MARGIN_PX = 0
PHOTODIODE_ON_COLOR = (255, 255, 255)
PHOTODIODE_OFF_COLOR = (0, 0, 0)

USE_GPIO = False
TTL_PIN_BCM = 23
TTL_PULSE_SEC = 0.005

EVENT_FIELDS = [
    "utc_iso",
    "unix_time_utc_sec",
    "event_type",
    "trial_index",
    "repeat_number",
    "image_index",
    "image_id",
    "image_filename",
    "raw_path",
    "planned_duration_sec",
    "display_request_unix_ns",
    "display_return_unix_ns",
    "display_return_utc_iso",
    "display_request_perf_counter_ns",
    "display_return_perf_counter_ns",
    "display_call_duration_sec",
    "start_time_unix",
    "mean_interframe_us",
    "stddev_interframe_us",
    "notes",
]


def sanitize_text(text):
    text = text.strip()
    cleaned = []
    for char in text:
        if char.isalnum() or char in "-_":
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned)


def unix_ns_to_iso(unix_ns):
    """Convert integer Unix nanoseconds to an ISO-8601 UTC string."""
    unix_ns = int(unix_ns)
    seconds, nanoseconds = divmod(unix_ns, 1_000_000_000)
    dt = datetime.fromtimestamp(seconds, timezone.utc)
    return "%s.%09d+00:00" % (
        dt.strftime("%Y-%m-%dT%H:%M:%S"),
        nanoseconds,
    )


def unix_ns_to_seconds_string(unix_ns):
    """Return an exact decimal Unix-seconds string with nine digits."""
    unix_ns = int(unix_ns)
    seconds, nanoseconds = divmod(unix_ns, 1_000_000_000)
    return "%d.%09d" % (seconds, nanoseconds)


def capture_timestamp():
    """Capture one absolute UTC timestamp and return all representations."""
    unix_ns = time.time_ns()
    return {
        "unix_ns": unix_ns,
        "unix_sec": unix_ns_to_seconds_string(unix_ns),
        "utc_iso": unix_ns_to_iso(unix_ns),
    }


def utc_iso_now():
    return unix_ns_to_iso(time.time_ns())


def utc_session_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_session_name(mouse_id, session_stamp):
    return "%s_%s_%s" % (mouse_id, session_stamp, SESSION_NAME_SUFFIX)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_image_dir():
    for candidate in IMAGE_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    checked = chr(10).join("  - %s" % candidate for candidate in IMAGE_DIR_CANDIDATES)
    raise RuntimeError(
        "Could not find the Stringer PNG folder. Checked:" + chr(10) + checked + chr(10) +
        "Place stringer_natimg2800_center_crop_png somewhere predictable and update IMAGE_DIR_CANDIDATES if needed."
    )

def list_png_files(image_dir):
    files = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    if not files:
        raise RuntimeError("No PNG files found in %s" % image_dir)
    return files


def parse_image_id_from_filename(path):
    for part in reversed(path.stem.split("_")):
        if part.isdigit():
            return int(part)
    return None


def select_image_subset(all_image_files, n_images_to_use):
    if n_images_to_use is None:
        return list(all_image_files)
    if n_images_to_use > len(all_image_files):
        raise RuntimeError(
            "N_IMAGES_TO_USE=%d, but only %d PNGs were found." % (n_images_to_use, len(all_image_files))
        )
    rng = random.Random(IMAGE_SUBSET_SEED)
    return sorted(rng.sample(list(all_image_files), n_images_to_use))


def make_trial_sequence(selected_image_files, n_repeats):
    if TRIAL_ORDER_SEED is None:
        seed = int(time.time_ns() % (2 ** 32))
    else:
        seed = int(TRIAL_ORDER_SEED)

    rng = random.Random(seed)
    base_trials = []
    for repeat_idx in range(n_repeats):
        for selected_index, path in enumerate(selected_image_files):
            image_id = parse_image_id_from_filename(path)
            base_trials.append(
                {
                    "image_index": selected_index,
                    "image_id": image_id if image_id is not None else selected_index,
                    "image_filename": path.name,
                    "image_path": str(path),
                    "repeat_number": repeat_idx + 1,
                }
            )

    if AVOID_ADJACENT_REPEATS:
        for _ in range(1000):
            rng.shuffle(base_trials)
            if all(base_trials[i]["image_id"] != base_trials[i - 1]["image_id"] for i in range(1, len(base_trials))):
                break
        else:
            print("Warning: could not fully avoid adjacent repeated images.")
    else:
        rng.shuffle(base_trials)

    trials = []
    for trial_index, trial in enumerate(base_trials):
        row = dict(trial)
        row["trial_index"] = trial_index
        row["planned_stim_duration_sec"] = STIM_DURATION_SEC
        row["planned_iti_duration_sec"] = ITI_DURATION_SEC
        trials.append(row)
    return trials, seed


def write_csv(path, rows, fieldnames):
    ensure_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def append_csv_row(path, row, fieldnames):
    ensure_dir(path.parent)
    file_exists = path.exists()
    row = dict(row)
    if not row.get("utc_iso") or not row.get("unix_time_utc_sec"):
        captured = capture_timestamp()
        row["utc_iso"] = captured["utc_iso"]
        row["unix_time_utc_sec"] = captured["unix_sec"]
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def refreshes_for_seconds(seconds):
    return max(1, int(round(seconds * REFRESH_RATE_HZ)))


def patch_rect(screen_size):
    width, _ = screen_size
    left = width - PHOTODIODE_SIZE_PX - PHOTODIODE_MARGIN_PX
    top = PHOTODIODE_MARGIN_PX
    return left, top, left + PHOTODIODE_SIZE_PX, top + PHOTODIODE_SIZE_PX


def build_canvas(image_path, screen_size, photodiode_on):
    canvas = Image.new("RGB", screen_size, (SCREEN_BACKGROUND_GRAY,) * 3)

    if image_path is not None:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = ImageOps.fit(img, screen_size, method=Image.LANCZOS, centering=(0.5, 0.5))
            canvas.paste(img, (0, 0))

    if ENABLE_PHOTODIODE_PATCH:
        draw = ImageDraw.Draw(canvas)
        fill = PHOTODIODE_ON_COLOR if photodiode_on else PHOTODIODE_OFF_COLOR
        draw.rectangle(patch_rect(screen_size), fill=fill)

    return canvas


def convert_canvas_to_rpg_raw(rpg_module, canvas, raw_path, duration_sec):
    ensure_dir(raw_path.parent)
    fd, source_name = tempfile.mkstemp(prefix="%s_" % raw_path.stem, suffix=".rgb.raw", dir=str(raw_path.parent))
    source_path = Path(source_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canvas.convert("RGB").tobytes())
        rpg_module.convert_raw(
            str(source_path),
            str(raw_path),
            1,
            SCREEN_RESOLUTION[0],
            SCREEN_RESOLUTION[1],
            refreshes_for_seconds(duration_sec),
            SCREEN_COLORMODE,
        )
    finally:
        if source_path.exists():
            try:
                source_path.unlink()
            except OSError:
                pass
    return raw_path


def display_raw_with_timing(screen, raw):
    """
    Call the blocking RPG display function while recording request and return timestamps.

    The request timestamp is captured immediately before display_raw().
    It represents the RPi software request, not physical monitor onset.
    """
    request = capture_timestamp()
    request_perf_ns = time.perf_counter_ns()

    perf = screen.display_raw(raw)

    return_perf_ns = time.perf_counter_ns()
    returned = capture_timestamp()

    return perf, {
        "request_unix_ns": request["unix_ns"],
        "request_unix_sec": request["unix_sec"],
        "request_utc_iso": request["utc_iso"],
        "return_unix_ns": returned["unix_ns"],
        "return_unix_sec": returned["unix_sec"],
        "return_utc_iso": returned["utc_iso"],
        "request_perf_counter_ns": request_perf_ns,
        "return_perf_counter_ns": return_perf_ns,
        "duration_sec": (return_perf_ns - request_perf_ns) / 1_000_000_000.0,
    }


def make_display_event_row(event_type, trial, raw_path, planned_duration_sec, perf, timing):
    return {
        "utc_iso": timing["request_utc_iso"],
        "unix_time_utc_sec": timing["request_unix_sec"],
        "event_type": event_type,
        "trial_index": trial["trial_index"],
        "repeat_number": trial["repeat_number"],
        "image_index": trial["image_index"],
        "image_id": trial["image_id"],
        "image_filename": trial["image_filename"],
        "raw_path": str(raw_path),
        "planned_duration_sec": planned_duration_sec,
        "display_request_unix_ns": timing["request_unix_ns"],
        "display_return_unix_ns": timing["return_unix_ns"],
        "display_return_utc_iso": timing["return_utc_iso"],
        "display_request_perf_counter_ns": timing["request_perf_counter_ns"],
        "display_return_perf_counter_ns": timing["return_perf_counter_ns"],
        "display_call_duration_sec": "%.9f" % timing["duration_sec"],
        "start_time_unix": getattr(perf, "start_time", ""),
        "mean_interframe_us": getattr(perf, "mean_interframe", ""),
        "stddev_interframe_us": getattr(perf, "stddev_interframe", ""),
        "notes": "",
    }


def run_trial_sequence(screen, trials, loaded_stim_raws, iti_raw, stim_raw_paths, iti_raw_path, event_log_path, gpio=None, include_final_iti=True):
    playback_start = time.perf_counter()
    total_trials = len(trials)
    for trial_index, trial in enumerate(trials, start=1):
        stem = Path(trial["image_path"]).stem
        if gpio is not None:
            ttl_pulse(gpio)

        stim_perf, stim_timing = display_raw_with_timing(screen, loaded_stim_raws[stem])

        stim_row = make_display_event_row(
            "stim_on",
            trial,
            stim_raw_paths[stem],
            STIM_DURATION_SEC,
            stim_perf,
            stim_timing,
        )
        append_csv_row(event_log_path, stim_row, EVENT_FIELDS)

        is_final_trial = trial_index == total_trials
        if include_final_iti or not is_final_trial:
            iti_perf, iti_timing = display_raw_with_timing(screen, iti_raw)
            iti_row = make_display_event_row(
                "iti_on",
                trial,
                iti_raw_path,
                ITI_DURATION_SEC,
                iti_perf,
                iti_timing,
            )
            append_csv_row(event_log_path, iti_row, EVENT_FIELDS)

        print_progress(trial_index, total_trials, playback_start)


def build_stim_raw_cache(rpg_module, session_raw_dir, selected_image_files):
    stim_dir = ensure_dir(session_raw_dir / "stim_on")

    stim_raw_paths = {}
    for image_path in selected_image_files:
        stim_canvas = build_canvas(image_path, SCREEN_RESOLUTION, photodiode_on=True)
        raw_path = stim_dir / (image_path.stem + ".raw")
        convert_canvas_to_rpg_raw(rpg_module, stim_canvas, raw_path, STIM_DURATION_SEC)
        stim_raw_paths[image_path.stem] = raw_path

    return stim_raw_paths


def build_iti_raw_cache(rpg_module, session_raw_dir):
    iti_dir = ensure_dir(session_raw_dir / "iti")

    iti_canvas = build_canvas(None, SCREEN_RESOLUTION, photodiode_on=False)
    iti_raw_path = iti_dir / "gray_iti.raw"
    convert_canvas_to_rpg_raw(rpg_module, iti_canvas, iti_raw_path, ITI_DURATION_SEC)
    return iti_raw_path


def build_session_raw_cache(rpg_module, session_raw_dir, selected_image_files):
    stim_raw_paths = build_stim_raw_cache(rpg_module, session_raw_dir, selected_image_files)
    iti_raw_path = build_iti_raw_cache(rpg_module, session_raw_dir)
    return stim_raw_paths, iti_raw_path

def setup_gpio():
    if not USE_GPIO:
        return None
    try:
        import RPi.GPIO as GPIO
    except ImportError as exc:
        raise RuntimeError("USE_GPIO=True, but RPi.GPIO could not be imported.") from exc

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TTL_PIN_BCM, GPIO.OUT)
    GPIO.output(TTL_PIN_BCM, GPIO.LOW)
    return GPIO


def ttl_pulse(GPIO):
    if GPIO is None:
        return
    GPIO.output(TTL_PIN_BCM, GPIO.HIGH)
    time.sleep(TTL_PULSE_SEC)
    GPIO.output(TTL_PIN_BCM, GPIO.LOW)


def print_environment():
    print("Display environment:")
    print("  DISPLAY=%s" % os.environ.get("DISPLAY"))
    print("  SDL_VIDEODRIVER=%s" % os.environ.get("SDL_VIDEODRIVER"))
    print("  XAUTHORITY=%s" % os.environ.get("XAUTHORITY"))


def prompt_text(prompt):
    try:
        return input(prompt)
    except EOFError:
        return ""


def prompt_int_or_default(prompt, default_value, minimum=1):
    raw = prompt_text("%s [%d]: " % (prompt, default_value)).strip()
    if not raw:
        return default_value
    value = int(raw)
    if value < minimum:
        raise ValueError("%s must be at least %d" % (prompt, minimum))
    return value


def prompt_yes_no(prompt, default_yes=True):
    suffix = "[Y/n]" if default_yes else "[y/N]"
    raw = prompt_text("%s %s: " % (prompt, suffix)).strip().lower()
    if not raw:
        return default_yes
    return raw in {"y", "yes"}


def format_seconds(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return "%02d:%02d:%02d" % (hours, minutes, secs)


def estimate_playback_seconds(total_trials):
    return INITIAL_GRAY_SEC + FINAL_GRAY_SEC + total_trials * (STIM_DURATION_SEC + ITI_DURATION_SEC)


def print_progress(trial_number, total_trials, start_time):
    elapsed = time.perf_counter() - start_time
    per_trial = STIM_DURATION_SEC + ITI_DURATION_SEC
    remaining_trials = max(0, total_trials - trial_number)
    estimated_remaining = remaining_trials * per_trial + FINAL_GRAY_SEC
    percent = 100.0 if total_trials == 0 else (trial_number / float(total_trials)) * 100.0
    line = "Progress: %d/%d (%.1f%%) ETA %s" % (trial_number, total_trials, percent, format_seconds(estimated_remaining))
    sys.stdout.write("\r" + line + " " * 8)
    sys.stdout.flush()
    return elapsed


def main():
    print_environment()
    try:
        import rpg
    except ImportError as exc:
        raise RuntimeError(
            "The rpg package is not installed. Install the SjulsonLab rpg repo on the behavior Pi first."
        ) from exc

    mouse_id_raw = prompt_text("Mouse ID: ")
    mouse_id = sanitize_text(mouse_id_raw) or "mouse"
    session_notes = prompt_text("Session notes, optional: ").strip()
    n_images_to_use = prompt_int_or_default("Number of unique images to use", N_IMAGES_TO_USE)
    n_repeats = prompt_int_or_default("Repeats per image", N_REPEATS)

    image_dir = resolve_image_dir()
    all_pngs = list_png_files(image_dir)
    selected_pngs = select_image_subset(all_pngs, n_images_to_use)
    trials, sequence_seed = make_trial_sequence(selected_pngs, n_repeats)
    estimated_playback_sec = estimate_playback_seconds(len(trials))

    print()
    print("Session setup summary:")
    print("  Mouse ID: %s" % (mouse_id_raw.strip() or mouse_id))
    print("  Session notes, optional: %s" % session_notes)
    print("  Number of unique images: %d" % len(selected_pngs))
    print("  Repeats per image: %d" % n_repeats)
    print("  Total trials: %d" % len(trials))
    print("  Estimated playback time: %s" % format_seconds(estimated_playback_sec))
    print("  Output folder root: %s" % OUTPUT_ROOT)
    print("  Session folder name: %s" % make_session_name(mouse_id, "YYYYMMDDThhmmssZ"))

    if not prompt_yes_no("Start this session", default_yes=True):
        print("Session aborted before starting. No files were changed.")
        return 0

    session_stamp = utc_session_stamp()
    session_id = make_session_name(mouse_id, session_stamp)

    session_root = ensure_dir(OUTPUT_ROOT / session_id)
    raw_cache_root = ensure_dir(session_root / "raw_cache")
    event_log_path = session_root / (session_id + "_event_log.csv")
    selected_images_path = session_root / (session_id + "_selected_images.csv")
    planned_sequence_path = session_root / (session_id + "_planned_sequence.csv")
    metadata_path = session_root / (session_id + "_metadata.json")

    print("Session ID: %s" % session_id)
    print("Selected images: %s" % selected_images_path)
    print("Planned sequence: %s" % planned_sequence_path)
    print("Event log: %s" % event_log_path)
    print("Metadata: %s" % metadata_path)
    selected_rows = []
    for index, image_path in enumerate(selected_pngs):
        image_id = parse_image_id_from_filename(image_path)
        selected_rows.append(
            {
                "selected_index": index,
                "image_id": image_id if image_id is not None else index,
                "image_filename": image_path.name,
                "image_path": str(image_path),
            }
        )
    write_csv(selected_images_path, selected_rows, ["selected_index", "image_id", "image_filename", "image_path"])

    write_csv(
        planned_sequence_path,
        [
            {
                "trial_index": trial["trial_index"],
                "image_index": trial["image_index"],
                "image_id": trial["image_id"],
                "image_filename": trial["image_filename"],
                "image_path": trial["image_path"],
                "repeat_number": trial["repeat_number"],
                "planned_stim_duration_sec": trial["planned_stim_duration_sec"],
                "planned_iti_duration_sec": trial["planned_iti_duration_sec"],
            }
            for trial in trials
        ],
        [
            "trial_index",
            "image_index",
            "image_id",
            "image_filename",
            "image_path",
            "repeat_number",
            "planned_stim_duration_sec",
            "planned_iti_duration_sec",
        ],
    )

    metadata = {
        "session_id": session_id,
        "utc_iso_start": utc_iso_now(),
        "mouse_id_input": mouse_id_raw,
        "mouse_id": mouse_id,
        "session_notes": session_notes,
        "image_dir": str(image_dir),
        "output_root": str(OUTPUT_ROOT),
        "screen_resolution": list(SCREEN_RESOLUTION),
        "screen_background_gray": SCREEN_BACKGROUND_GRAY,
        "screen_colormode": SCREEN_COLORMODE,
        "refresh_rate_hz": REFRESH_RATE_HZ,
        "n_images_to_use": n_images_to_use,
        "n_repeats": n_repeats,
        "image_subset_seed": IMAGE_SUBSET_SEED,
        "trial_order_seed": TRIAL_ORDER_SEED,
        "resolved_trial_order_seed": sequence_seed,
        "avoid_adjacent_repeats": AVOID_ADJACENT_REPEATS,
        "stim_duration_sec": STIM_DURATION_SEC,
        "iti_duration_sec": ITI_DURATION_SEC,
        "initial_gray_sec": INITIAL_GRAY_SEC,
        "final_gray_sec": FINAL_GRAY_SEC,
        "enable_photodiode_patch": ENABLE_PHOTODIODE_PATCH,
        "photodiode_size_px": PHOTODIODE_SIZE_PX,
        "photodiode_margin_px": PHOTODIODE_MARGIN_PX,
        "use_gpio": USE_GPIO,
        "ttl_pin_bcm": TTL_PIN_BCM,
        "selected_images": selected_rows,
        "trials": trials,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + chr(10))

    print("Preparing session raw files...")
    stim_raw_paths, iti_raw_path = build_session_raw_cache(rpg, raw_cache_root, selected_pngs)
    print("Session raw files ready. Starting visual stimulus playback...")
    sys.stdout.flush()
    loaded_stim_raws = {}

    gpio = None
    session_completed = False
    if USE_GPIO:
        gpio = setup_gpio()

    try:
        with rpg.Screen(SCREEN_RESOLUTION, background=SCREEN_BACKGROUND_GRAY, colormode=SCREEN_COLORMODE) as screen:
            screen.display_greyscale(SCREEN_BACKGROUND_GRAY)
            time.sleep(INITIAL_GRAY_SEC)

            for image_path in selected_pngs:
                loaded_stim_raws[image_path.stem] = screen.load_raw(str(stim_raw_paths[image_path.stem]))
            iti_raw = screen.load_raw(str(iti_raw_path))

            append_csv_row(
                event_log_path,
                {
                    "utc_iso": utc_iso_now(),
                    "event_type": "session_start",
                    "notes": "screen_opened",
                },
                EVENT_FIELDS,
            )
            print("Stimulus playback is now active.")
            sys.stdout.flush()

            run_trial_sequence(
                screen,
                trials,
                loaded_stim_raws,
                iti_raw,
                stim_raw_paths,
                iti_raw_path,
                event_log_path,
                gpio=gpio if USE_GPIO else None,
            )

            screen.display_greyscale(SCREEN_BACKGROUND_GRAY)
            time.sleep(FINAL_GRAY_SEC)
            sys.stdout.write("\n")
            sys.stdout.flush()
            append_csv_row(
                event_log_path,
                {
                    "utc_iso": utc_iso_now(),
                    "event_type": "session_end",
                    "notes": "completed",
                },
                EVENT_FIELDS,
            )
            session_completed = True

    except KeyboardInterrupt:
        append_csv_row(
            event_log_path,
            {
                "utc_iso": utc_iso_now(),
                "event_type": "session_end",
                "notes": "keyboard_interrupt",
            },
            EVENT_FIELDS,
        )
        raise
    finally:
        if gpio is not None:
            try:
                import RPi.GPIO as GPIO
                GPIO.output(TTL_PIN_BCM, GPIO.LOW)
                GPIO.cleanup()
            except Exception:
                pass
        metadata["utc_iso_end"] = utc_iso_now()
        metadata["event_log"] = str(event_log_path)
        metadata["selected_images_csv"] = str(selected_images_path)
        metadata["planned_sequence_csv"] = str(planned_sequence_path)
        metadata["raw_cache_root"] = str(raw_cache_root)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + chr(10))
        if session_completed:
            print("Session finished. Files are in: %s" % session_root)
        else:
            print("Session stopped early. Partial files are in: %s" % session_root)
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    print(
        "This repository's supported experiment entrypoint is:\n\n"
        "  python3 run_stringer_reward_conditioning.py\n\n"
        "For the natural-image-only protocol use:\n"
        "  https://github.com/hung-lo/vstim_natural",
        file=sys.stderr,
    )
    raise SystemExit(2)
