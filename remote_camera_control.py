#!/usr/bin/env python3
"""
remote_camera_control.py

Standalone helper for controlling the second Raspberry Pi camera without
touching run_stringer_vstim.py.

Camera Pi is hard-coded here:
    pi@192.168.1.152

Typical use on the behavior Pi:

    cd /home/pi/vstim_natural_img_reward
    source .venv/bin/activate

    python3 remote_camera_control.py start --mouse-id testmouse
    python3 remote_camera_control.py preview
    python3 remote_camera_control.py status
    python3 remote_camera_control.py stop-fetch

You can still override the host manually with:
    --camera-host pi@OTHER_IP
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_CAMERA_HOST = "pi@192.168.1.152"

REMOTE_CAMERA_REPO = "/home/pi/RPi4_behavior_boxes"
REMOTE_CAMERA_START = "/home/pi/RPi4_behavior_boxes/video_acquisition/start_acquisition.py"
REMOTE_CAMERA_STOP = "/home/pi/RPi4_behavior_boxes/video_acquisition/stop_acquisition.sh"
REMOTE_CAMERA_PREVIEW_LOG = "/home/pi/stim_logs/camera_preview.log"
REMOTE_CAMERA_PREVIEW_PID_FILE = "/tmp/remote_camera_preview.pid"

REMOTE_VIDEO_ROOT = "/home/pi/stim_logs"
LOCAL_VIDEO_ROOT = Path("/mnt/hd")

SSH_OPTIONS = [
    "-n",
    "-T",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=5",
    "-o", "ServerAliveInterval=2",
    "-o", "ServerAliveCountMax=3",
]
CAMERA_START_LAUNCH_TIMEOUT_SEC = 10.0
CAMERA_START_READY_TIMEOUT_SEC = 8.0
CAMERA_READY_POLL_SEC = 0.25
CAMERA_LOG_TAIL_LINES = 50
CAMERA_OUTPUT_READY_TIMEOUT_SEC = 12.0
CAMERA_OUTPUT_READY_POLL_SEC = 0.5
CAMERA_OUTPUT_MIN_BYTES = 1
CAMERA_OUTPUT_GLOB = "*.h264"
CAMERA_RAW_VIDEO_PATTERNS = ("*.h264",)
CAMERA_RSYNC_TIMEOUT_SEC = 300.0
CAMERA_MANIFEST_TIMEOUT_SEC = 300.0
CAMERA_FFMPEG_TIMEOUT_SEC = 120.0
CAMERA_STOP_TIMEOUT_SEC = 10.0
CAMERA_PREVIEW_LAUNCH_TIMEOUT_SEC = 10.0
CAMERA_STOP_VERIFY_TIMEOUT_SEC = 8.0
CAMERA_STOP_VERIFY_POLL_SEC = 0.25
CAMERA_PREVIEW_STOP_TIMEOUT_SEC = 5.0
SESSION_NAME_SUFFIX = "vstim_natural_img_reward"
CAMERA_STATE_SCHEMA_VERSION = 1
CONTROLLER_REPOSITORY = "vstim_natural_img_reward"
PROTOCOL_NAME = "open_loop_natural_image_reward_conditioning"

CAMERA_FRAMERATE = 30
JSON_RESULT_PREFIX = "CAMERA_CONTROL_RESULT_JSON="

PROJECT_ROOT = Path(__file__).resolve().parent
STATE_FILE = LOCAL_VIDEO_ROOT / ".vstim_natural_img_reward_camera_session.json"
LEGACY_STATE_FILE = LOCAL_VIDEO_ROOT / ".last_remote_camera_session.json"


class CameraStateError(RuntimeError):
    pass


class CameraStateNotFoundError(CameraStateError):
    pass


class CameraStateOwnershipError(CameraStateError):
    pass


class CameraStateSchemaError(CameraStateError):
    pass


class CameraStateCorruptError(CameraStateError):
    pass


def utc_iso_now():
    return datetime.now(timezone.utc).isoformat()


def utc_label():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_session_name(mouse_id, session_stamp):
    return "%s_%s_%s" % (mouse_id, session_stamp, SESSION_NAME_SUFFIX)


def sanitize_id(text):
    text = str(text).strip()
    keep = []
    for char in text:
        if char.isalnum() or char in ["-", "_"]:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)


def subprocess_output_to_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def tail_text(text, max_chars=4000):
    text = text or ""
    if len(text) <= max_chars:
        return text
    return "...[truncated]...\n" + text[-max_chars:]


def print_subprocess_output(stdout, stderr):
    stdout_text = subprocess_output_to_text(stdout)
    stderr_text = subprocess_output_to_text(stderr)
    if stdout_text:
        print(stdout_text, end="" if stdout_text.endswith("\n") else "\n")
    if stderr_text:
        print(stderr_text, end="" if stderr_text.endswith("\n") else "\n", file=sys.stderr)
    return stdout_text, stderr_text


def run_cmd(cmd, check=True, dry_run=False, timeout=None):
    print("+ " + " ".join(shlex.quote(x) for x in cmd))
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    try:
        return subprocess.run(cmd, check=check, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stdout_text, stderr_text = print_subprocess_output(exc.stdout, exc.stderr)
        timeout_desc = "unknown" if timeout is None else "%.1f" % timeout
        detail_lines = []
        if stdout_text:
            detail_lines.append("stdout tail:\n%s" % tail_text(stdout_text))
        if stderr_text:
            detail_lines.append("stderr tail:\n%s" % tail_text(stderr_text))
        message = "Command timed out after %s seconds: %s" % (timeout_desc, " ".join(shlex.quote(x) for x in cmd))
        if detail_lines:
            message += "\n" + "\n".join(detail_lines)
        raise RuntimeError(message) from exc
    except subprocess.CalledProcessError as exc:
        print_subprocess_output(exc.stdout, exc.stderr)
        raise


def run_ssh(camera_host, remote_cmd, check=True, dry_run=False, timeout=None):
    return run_cmd(["ssh", *SSH_OPTIONS, camera_host, remote_cmd], check=check, dry_run=dry_run, timeout=timeout)


def build_remote_camera_launch_command(paths, framerate, remote_camera_repo, remote_camera_start, remote_log, pid_file):
    return (
        "set -e; "
        f"mkdir -p {shlex.quote(paths['remote_video_dir'])}; "
        f"cd {shlex.quote(remote_camera_repo)}; "
        f"nohup python3 {shlex.quote(remote_camera_start)} {shlex.quote(paths['remote_base_path'])} {int(framerate)} </dev/null >> {shlex.quote(remote_log)} 2>&1 & "
        "pid=$!; "
        f"echo \"$pid\" > {shlex.quote(pid_file)}; "
        "printf 'CAMERA_PID=%s\\n' \"$pid\""
    )


def build_remote_pid_check_command(pid_file):
    return (
        f"pid=$(cat {shlex.quote(pid_file)} 2>/dev/null || true); "
        "if [ -n \"$pid\" ] && kill -0 \"$pid\" 2>/dev/null; then "
        "echo \"RUNNING:$pid\"; "
        "exit 0; "
        "fi; "
        "echo NOT_RUNNING; "
        "exit 1"
    )


def build_remote_log_tail_command(log_file, n_lines=CAMERA_LOG_TAIL_LINES):
    return "tail -n %d %s 2>/dev/null || true" % (int(n_lines), shlex.quote(log_file))


def build_remote_camera_stop_command(pid_file, remote_stop_script):
    return (
        "set -e; "
        f"pid=$(cat {shlex.quote(pid_file)} 2>/dev/null || true); "
        "if [ -n \"$pid\" ] && [ -r /proc/$pid/cmdline ]; then "
        "cmdline=$(tr '\\0' ' ' < /proc/$pid/cmdline); "
        "case \"$cmdline\" in "
        "*start_acquisition.py*) "
        "kill \"$pid\" 2>/dev/null || true; "
        "sleep 1; "
        "kill -9 \"$pid\" 2>/dev/null || true; "
        ";; "
        "*) echo 'PID file does not match expected acquisition process; falling back to stop script' >&2; ;; "
        "esac; "
        "fi; "
        f"bash {shlex.quote(remote_stop_script)}"
    )



def build_remote_camera_stopped_check_command(expected_pid, remote_base_path):
    expected_pid_text = "" if expected_pid is None else str(int(expected_pid))
    return (
        "pid_alive=0; "
        "expected_pid=%s; "
        "if [ -n \"$expected_pid\" ] && kill -0 \"$expected_pid\" 2>/dev/null; then pid_alive=1; fi; "
        "matches=$(pgrep -af '[s]tart_acquisition.py' | grep -F -- %s || true); "
        "if [ \"$pid_alive\" -eq 0 ] && [ -z \"$matches\" ]; then "
        "echo STOPPED; exit 0; "
        "fi; "
        "echo STILL_RUNNING; "
        "if [ -n \"$matches\" ]; then echo \"$matches\"; fi; "
        "exit 1"
        % (shlex.quote(expected_pid_text), shlex.quote(remote_base_path))
    )


def build_remote_output_probe_command(remote_video_dir):
    return (
        "find %s -maxdepth 1 -type f -name %s "
        "-printf '%%p\\t%%s\\n' 2>/dev/null | sort"
        % (shlex.quote(remote_video_dir), shlex.quote(CAMERA_OUTPUT_GLOB))
    )


def build_remote_raw_manifest_command(remote_video_dir):
    """Return a tab-delimited manifest for this session directory only."""
    return (
        "find %s -maxdepth 1 -type f -name '*.h264' -exec sh -c '"
        "for file do size=$(stat -c %%s \"$file\"); set -- $(sha256sum \"$file\"); "
        "hash=$1; printf \"%%s\\t%%s\\t%%s\\n\" \"$file\" \"$size\" \"$hash\"; done' sh {} +"
        % shlex.quote(remote_video_dir)
    )


def parse_camera_pid(stdout):
    match = re.search(r"CAMERA_PID=(\d+)", stdout or "")
    if match:
        return int(match.group(1))
    match = re.search(r"RUNNING:(\d+)", stdout or "")
    if match:
        return int(match.group(1))
    return None


def parse_remote_output_probe(stdout):
    files = {}
    for line in (stdout or "").splitlines():
        try:
            path, size_text = line.rsplit("\t", 1)
            files[path] = int(size_text)
        except (TypeError, ValueError):
            continue
    return files


def parse_remote_raw_manifest(stdout):
    manifest = []
    for line in (stdout or "").splitlines():
        try:
            remote_path, size_text, sha256 = line.rsplit("\t", 2)
            size_bytes = int(size_text)
        except (TypeError, ValueError):
            continue
        if size_bytes < 0 or not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
            continue
        manifest.append({"remote_path": remote_path, "filename": Path(remote_path).name,
                         "size_bytes": size_bytes, "sha256": sha256.lower()})
    return manifest


def tail_remote_log(camera_host, log_file, dry_run=False):
    result = run_ssh(
        camera_host,
        build_remote_log_tail_command(log_file),
        check=False,
        dry_run=dry_run,
        timeout=CAMERA_START_READY_TIMEOUT_SEC,
    )
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    return (stdout + stderr).strip()


def wait_for_remote_camera_ready(camera_host, pid_file, log_file, dry_run=False, timeout_sec=CAMERA_START_READY_TIMEOUT_SEC):
    if dry_run:
        return {"camera_pid": None, "stdout": "[dry-run]", "returncode": 0}

    deadline = time.monotonic() + float(timeout_sec)
    last_output = ""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        poll_timeout = min(CAMERA_READY_POLL_SEC, max(0.2, remaining))
        result = run_ssh(
            camera_host,
            build_remote_pid_check_command(pid_file),
            check=False,
            dry_run=dry_run,
            timeout=poll_timeout + 2.0,
        )
        last_output = (result.stdout or "") + (result.stderr or "")
        if result.returncode == 0:
            pid = parse_camera_pid(last_output)
            return {"camera_pid": pid, "stdout": last_output, "returncode": result.returncode}
        time.sleep(min(CAMERA_READY_POLL_SEC, max(0.0, remaining)))

    tail = tail_remote_log(camera_host, log_file, dry_run=dry_run)
    raise RuntimeError(
        "Camera did not report ready within %.1f seconds. Last remote log tail:\n%s"
        % (timeout_sec, tail or last_output or "<no output>")
    )


def wait_for_remote_camera_output_growth(
    camera_host,
    pid_file,
    remote_video_dir,
    log_file,
    dry_run=False,
    timeout_sec=CAMERA_OUTPUT_READY_TIMEOUT_SEC,
):
    if dry_run:
        return {
            "camera_output_file": None,
            "initial_size_bytes": 0,
            "confirmed_size_bytes": CAMERA_OUTPUT_MIN_BYTES,
            "camera_output_file_detected": True,
            "camera_output_growing_confirmed": True,
        }

    deadline = time.monotonic() + float(timeout_sec)
    first_sizes = {}
    last_output = ""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        pid_result = run_ssh(
            camera_host,
            build_remote_pid_check_command(pid_file),
            check=False,
            dry_run=dry_run,
            timeout=min(2.0, max(0.2, remaining)) + 2.0,
        )
        if pid_result.returncode != 0:
            tail = tail_remote_log(camera_host, log_file, dry_run=dry_run)
            raise RuntimeError(
                "Camera process stopped while waiting for video output growth. "
                "Last remote log tail:\n%s" % (tail or "<no output>")
            )

        probe_result = run_ssh(
            camera_host,
            build_remote_output_probe_command(remote_video_dir),
            check=False,
            dry_run=dry_run,
            timeout=min(2.0, max(0.2, remaining)) + 2.0,
        )
        last_output = (probe_result.stdout or "") + (probe_result.stderr or "")
        current_sizes = parse_remote_output_probe(probe_result.stdout)
        for path, current_size in current_sizes.items():
            if path not in first_sizes:
                first_sizes[path] = current_size
                continue
            initial_size = first_sizes[path]
            if current_size >= CAMERA_OUTPUT_MIN_BYTES and current_size > initial_size:
                return {
                    "camera_output_file": path,
                    "initial_size_bytes": initial_size,
                    "confirmed_size_bytes": current_size,
                    "camera_output_file_detected": True,
                    "camera_output_growing_confirmed": True,
                }

        time.sleep(min(CAMERA_OUTPUT_READY_POLL_SEC, max(0.0, remaining)))

    tail = tail_remote_log(camera_host, log_file, dry_run=dry_run)
    detected = bool(first_sizes)
    raise RuntimeError(
        "Camera video output %s within %.1f seconds. Last probe output:\n%s\n"
        "Last remote log tail:\n%s"
        % (
            "did not grow" if detected else "was not detected",
            timeout_sec,
            last_output or "<no output>",
            tail or "<no output>",
        )
    )


def wait_for_remote_camera_stopped(
    camera_host,
    expected_pid,
    remote_base_path,
    log_file,
    dry_run=False,
    timeout_sec=CAMERA_STOP_VERIFY_TIMEOUT_SEC,
):
    if dry_run:
        return {"camera_stop_confirmed": True, "stdout": "[dry-run]"}

    deadline = time.monotonic() + float(timeout_sec)
    last_output = ""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        result = run_ssh(
            camera_host,
            build_remote_camera_stopped_check_command(expected_pid, remote_base_path),
            check=False,
            dry_run=dry_run,
            timeout=min(2.0, max(0.2, remaining)) + 2.0,
        )
        last_output = (result.stdout or "") + (result.stderr or "")
        if result.returncode == 0:
            return {"camera_stop_confirmed": True, "stdout": last_output}
        time.sleep(min(CAMERA_STOP_VERIFY_POLL_SEC, max(0.0, remaining)))

    tail = tail_remote_log(camera_host, log_file, dry_run=dry_run)
    raise RuntimeError(
        "Camera stop could not be verified within %.1f seconds. "
        "Last process check:\n%s\nLast remote log tail:\n%s"
        % (timeout_sec, last_output or "<no output>", tail or "<no output>")
    )


def run_rsync(camera_host, remote_dir, local_dir, dry_run=False, timeout_sec=CAMERA_RSYNC_TIMEOUT_SEC):
    local_dir.mkdir(parents=True, exist_ok=True)
    return run_cmd(
        [
            "rsync",
            "-a",
            "--progress",
            "--partial",
            "--append-verify",
            "--timeout=30",
            f"{camera_host}:{remote_dir.rstrip('/')}/",
            str(local_dir) + "/",
        ],
        check=True,
        dry_run=dry_run,
        timeout=timeout_sec,
    )


def collect_remote_raw_manifest(camera_host, remote_video_dir, dry_run=False):
    if dry_run:
        return []
    result = run_ssh(camera_host, build_remote_raw_manifest_command(remote_video_dir),
                     dry_run=dry_run, timeout=CAMERA_MANIFEST_TIMEOUT_SEC)
    manifest = parse_remote_raw_manifest(result.stdout)
    if not manifest:
        raise RuntimeError("No session-owned .h264 files were found in %s." % remote_video_dir)
    return manifest


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_local_camera_raw_manifest(local_video_dir, manifest):
    local_video_dir = Path(local_video_dir)
    if not manifest:
        raise RuntimeError("Cannot verify local raw video without a remote SHA-256 manifest.")
    verified = []
    for item in manifest:
        local_path = local_video_dir / item["filename"]
        if not local_path.is_file():
            raise RuntimeError("Expected local raw video is missing: %s" % local_path)
        size_bytes = local_path.stat().st_size
        if size_bytes != int(item["size_bytes"]):
            raise RuntimeError("Local raw video size mismatch for %s: %d != %d" %
                               (local_path.name, size_bytes, item["size_bytes"]))
        local_hash = sha256_file(local_path)
        if local_hash.lower() != item["sha256"].lower():
            raise RuntimeError("Local raw video SHA-256 mismatch for %s" % local_path.name)
        verified.append(local_path)
    return verified


def write_video_manifest(local_video_dir, manifest):
    path = Path(local_video_dir) / "video_manifest.json"
    path.write_text(json.dumps({"remote_raw_manifest": manifest}, indent=2, sort_keys=True) + "\n")
    return path


def build_remote_cleanup_command(manifest, remote_video_dir):
    allowed_prefix = remote_video_dir.rstrip("/") + "/"
    paths = []
    for item in manifest:
        path = item["remote_path"]
        if not path.startswith(allowed_prefix) or Path(path).suffix != ".h264":
            raise RuntimeError("Refusing to clean a remote path outside this session: %s" % path)
        paths.append(path)
    if not paths:
        raise RuntimeError("No manifest paths were supplied for remote cleanup.")
    return "rm -f -- " + " ".join(shlex.quote(path) for path in paths)


def maybe_cleanup_remote_raw(args, state, camera_host, local_video_dir):
    """Freshly verify raw and MP4 data immediately before remote deletion."""
    state["remote_raw_cleanup_attempted"] = False
    manifest = state.get("remote_raw_manifest") or []
    state["camera_raw_files_verified"] = False
    state["camera_raw_hash_verified"] = False
    state["camera_mp4_verified"] = False
    state["remote_raw_cleanup_completed"] = False
    if not (state.get("camera_stop_confirmed") and manifest):
        state["remote_raw_retained"] = True
        return state
    try:
        verify_local_camera_raw_manifest(local_video_dir, manifest)
        state["camera_raw_files_verified"] = True
        state["camera_raw_hash_verified"] = True
        for item in manifest:
            probe_mp4(Path(local_video_dir) / Path(item["filename"]).with_suffix(".mp4"))
        state["camera_mp4_verified"] = True
        state["remote_raw_cleanup_attempted"] = True
        run_ssh(camera_host, build_remote_cleanup_command(manifest, state["remote_video_dir"]),
                dry_run=args.dry_run, timeout=CAMERA_STOP_TIMEOUT_SEC)
        state["remote_raw_cleanup_completed"] = True
        state["remote_raw_retained"] = False
        state["remote_raw_cleanup_error"] = ""
    except Exception as exc:
        # Raw verification leaves all current verification fields false; MP4
        # verification happens only after a current raw hash succeeds.
        state["remote_raw_cleanup_completed"] = False
        state["remote_raw_retained"] = True
        state["remote_raw_cleanup_error"] = str(exc)
        append_event(local_video_dir, "remote_raw_cleanup_failed", {"error": str(exc)})
    return state


def find_local_camera_raw_files(local_video_dir):
    raw_files = []
    seen = set()
    for pattern in CAMERA_RAW_VIDEO_PATTERNS:
        for path in sorted(local_video_dir.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            if path.is_file():
                raw_files.append(path)
    return raw_files


def verify_local_camera_raw_files(local_video_dir):
    raw_files = []
    empty_files = []
    for path in find_local_camera_raw_files(local_video_dir):
        size_bytes = path.stat().st_size
        if size_bytes > 0:
            raw_files.append(path)
        else:
            empty_files.append(path)

    if empty_files:
        raise RuntimeError(
            "Empty camera raw files were found after rsync completed: %s"
            % ", ".join(str(path) for path in empty_files)
        )
    if not raw_files:
        raise RuntimeError(
            "No .h264 camera files were found in the local video directory after rsync completed."
        )
    return raw_files


def probe_mp4(path, dry_run=False):
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is not installed or not available on PATH.")
    if dry_run:
        return {"codec_name": "h264", "width": 1, "height": 1, "duration_sec": 1.0, "frame_rate": "30/1"}
    result = run_cmd([ffprobe, "-v", "error", "-select_streams", "v:0",
                      "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate:format=duration",
                      "-of", "json", str(path)], timeout=CAMERA_FFMPEG_TIMEOUT_SEC)
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError("MP4 has no video stream: %s" % path)
    stream = streams[0]
    try:
        width, height, duration = int(stream.get("width", 0)), int(stream.get("height", 0)), float((payload.get("format") or {}).get("duration", 0))
    except (TypeError, ValueError):
        raise RuntimeError("MP4 has invalid dimensions or duration: %s" % path)
    frame_rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or ""
    rate_match = re.fullmatch(r"(\d+)\/(\d+)", str(frame_rate))
    if not rate_match or int(rate_match.group(1)) <= 0 or int(rate_match.group(2)) <= 0 or width <= 0 or height <= 0 or not (duration > 0):
        raise RuntimeError("MP4 failed ffprobe validation: %s" % path)
    return {"codec_name": stream.get("codec_name", ""), "width": width, "height": height,
            "duration_sec": duration, "frame_rate": frame_rate}


def preserve_invalid_mp4(path):
    path = Path(path)
    candidate = path.with_name(path.stem + ".invalid" + path.suffix)
    index = 1
    while candidate.exists():
        candidate = path.with_name(path.stem + ".invalid.%d" % index + path.suffix)
        index += 1
    path.rename(candidate)
    return candidate


def convert_h264_to_mp4(local_video_dir, framerate=CAMERA_FRAMERATE, dry_run=False, timeout_sec=CAMERA_FFMPEG_TIMEOUT_SEC):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg is not installed or not available on PATH; raw .h264 files were fetched but MP4 conversion could not run."
        )

    h264_files = verify_local_camera_raw_files(local_video_dir)
    result = {
        "conversion_attempted": False,
        "conversion_completed": False,
        "conversion_skipped": False,
        "conversion_skip_reason": None,
        "input_files": [str(path) for path in h264_files],
        "output_files": [],
        "mp4_probe": {},
    }

    if dry_run:
        result["conversion_skipped"] = True
        result["conversion_skip_reason"] = "dry_run"
        return result

    for input_path in h264_files:
        output_path = input_path.with_suffix(".mp4")
        if output_path.exists():
            try:
                result["mp4_probe"][str(output_path)] = probe_mp4(output_path, dry_run=dry_run)
                print("MP4 already valid, skipping: %s" % output_path)
                result["output_files"].append(str(output_path))
                continue
            except Exception:
                preserved = preserve_invalid_mp4(output_path)
                print("Existing invalid MP4 preserved as: %s" % preserved, file=sys.stderr)

        result["conversion_attempted"] = True
        part_path = output_path.with_name(output_path.name + ".part")
        if part_path.exists():
            part_path.unlink()
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(framerate),
            "-i",
            str(input_path),
            "-c:v",
            "copy",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(part_path),
        ]
        print("+ " + " ".join(shlex.quote(x) for x in cmd))
        try:
            subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            if part_path.exists():
                part_path.unlink()
            print_subprocess_output(exc.stdout, exc.stderr)
            raise RuntimeError("MP4 conversion timed out after %.1f seconds for %s" % (timeout_sec, input_path.name)) from exc
        except subprocess.CalledProcessError as exc:
            if part_path.exists():
                part_path.unlink()
            print_subprocess_output(exc.stdout, exc.stderr)
            raise

        if not part_path.exists() or part_path.stat().st_size <= 0:
            raise RuntimeError("FFmpeg did not create a nonempty temporary MP4: %s" % part_path)
        try:
            result["mp4_probe"][str(output_path)] = probe_mp4(part_path, dry_run=dry_run)
        except Exception:
            if part_path.exists():
                part_path.unlink()
            raise
        os.replace(str(part_path), str(output_path))

        result["output_files"].append(str(output_path))
        print("Converted %s -> %s" % (input_path.name, output_path.name))

    result["conversion_completed"] = bool(result["output_files"]) and len(result["output_files"]) == len(h264_files)
    return result


def run_camera_conversion_workflow(camera_host, local_video_dir, framerate=CAMERA_FRAMERATE, dry_run=False, timeout_sec=CAMERA_FFMPEG_TIMEOUT_SEC):
    local_video_dir = Path(local_video_dir)
    result = {
        "camera_raw_files_verified": False,
        "camera_raw_file_count": 0,
        "raw_h264_available_locally": False,
        "camera_conversion_attempted": False,
        "camera_conversion_completed": False,
        "camera_conversion_deferred": False,
        "camera_conversion_error": "",
        "camera_conversion_skip_reason": "",
        "converted_mp4_files": [],
        "conversion_result": None,
    }

    try:
        raw_files = verify_local_camera_raw_files(local_video_dir)
    except Exception as exc:
        append_event(
            local_video_dir,
            "camera_raw_verification_failed",
            {
                "camera_host": camera_host,
                "local_video_dir": str(local_video_dir),
                "error": str(exc),
            },
        )
        raise

    result["camera_raw_files_verified"] = True
    result["camera_raw_file_count"] = len(raw_files)
    result["raw_h264_available_locally"] = True
    append_event(
        local_video_dir,
        "camera_raw_files_verified",
        {
            "camera_host": camera_host,
            "local_video_dir": str(local_video_dir),
            "raw_file_count": len(raw_files),
            "raw_files": [str(path) for path in raw_files],
        },
    )

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        error = (
            "FFmpeg is not installed or not available on PATH; raw .h264 files were fetched but MP4 conversion could not run."
        )
        result["camera_conversion_deferred"] = True
        result["camera_conversion_error"] = error
        result["camera_conversion_skip_reason"] = "ffmpeg_not_available"
        append_event(
            local_video_dir,
            "camera_conversion_deferred",
            {
                "camera_host": camera_host,
                "local_video_dir": str(local_video_dir),
                "reason": "ffmpeg_not_available",
                "raw_h264_available_locally": True,
                "error": error,
            },
        )
        print(error, file=sys.stderr)
        return result

    append_event(
        local_video_dir,
        "camera_conversion_started",
        {
            "camera_host": camera_host,
            "local_video_dir": str(local_video_dir),
            "ffmpeg_timeout_sec": timeout_sec,
            "raw_file_count": len(raw_files),
        },
    )

    try:
        conversion_result = convert_h264_to_mp4(
            local_video_dir,
            framerate=framerate,
            dry_run=dry_run,
            timeout_sec=timeout_sec,
        )
    except Exception as exc:
        result["camera_conversion_attempted"] = True
        result["camera_conversion_deferred"] = True
        result["camera_conversion_error"] = str(exc)
        append_event(
            local_video_dir,
            "camera_conversion_failed",
            {
                "camera_host": camera_host,
                "local_video_dir": str(local_video_dir),
                "ffmpeg_timeout_sec": timeout_sec,
                "error": str(exc),
            },
        )
        print("MP4 conversion did not complete.", file=sys.stderr)
        return result

    result["conversion_result"] = conversion_result
    result["camera_conversion_attempted"] = bool(conversion_result.get("conversion_attempted"))
    result["camera_conversion_completed"] = bool(conversion_result.get("conversion_completed"))
    result["camera_conversion_deferred"] = bool(conversion_result.get("conversion_skipped"))
    result["camera_conversion_skip_reason"] = conversion_result.get("conversion_skip_reason") or ""
    result["converted_mp4_files"] = list(conversion_result.get("output_files", []))

    if result["camera_conversion_completed"]:
        append_event(
            local_video_dir,
            "camera_conversion_completed",
            {
                "camera_host": camera_host,
                "local_video_dir": str(local_video_dir),
                "converted_mp4_files": result["converted_mp4_files"],
                "raw_file_count": len(raw_files),
            },
        )
    else:
        append_event(
            local_video_dir,
            "camera_conversion_deferred",
            {
                "camera_host": camera_host,
                "local_video_dir": str(local_video_dir),
                "reason": result["camera_conversion_skip_reason"] or "dry_run",
                "raw_h264_available_locally": True,
            },
        )

    return result


def append_event(local_video_dir, event, details=None):
    local_video_dir.mkdir(parents=True, exist_ok=True)
    path = local_video_dir / "camera_control_events.csv"
    exists = path.exists()

    fieldnames = ["unix_time_utc_sec", "iso_time_utc", "event", "details_json"]
    row = {
        "unix_time_utc_sec": "%.6f" % time.time(),
        "iso_time_utc": utc_iso_now(),
        "event": event,
        "details_json": json.dumps(details or {}, sort_keys=True),
    }

    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state.setdefault("state_schema_version", CAMERA_STATE_SCHEMA_VERSION)
    state.setdefault("controller_repository", CONTROLLER_REPOSITORY)
    state.setdefault("protocol_name", PROTOCOL_NAME)
    temporary = STATE_FILE.with_name(".%s.%s.tmp" % (STATE_FILE.name, uuid.uuid4().hex))
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(STATE_FILE))
    finally:
        if temporary.exists():
            temporary.unlink()
    print("Saved state: %s" % STATE_FILE)


def load_state():
    if not STATE_FILE.exists():
        raise CameraStateNotFoundError(
            "No saved camera session state found at %s.\n"
            "A legacy state file may exist at %s, but this reward-conditioning controller will not operate on it.\n"
            "Run `python3 remote_camera_control.py start --mouse-id <mouse_id>` first, "
            "or pass `--mouse-id` and `--session-id` to fetch/stop-fetch."
            % (STATE_FILE, LEGACY_STATE_FILE)
        )
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise CameraStateCorruptError("Camera state is corrupt at %s: %s" % (STATE_FILE, exc)) from exc
    if state.get("state_schema_version") != CAMERA_STATE_SCHEMA_VERSION:
        raise CameraStateSchemaError("Camera state schema at %s is %r; expected %r."
                                     % (STATE_FILE, state.get("state_schema_version"), CAMERA_STATE_SCHEMA_VERSION))
    if state.get("controller_repository") != CONTROLLER_REPOSITORY:
        raise CameraStateOwnershipError("Camera state at %s belongs to %r, not %s; refusing to operate on it."
                           % (STATE_FILE, state.get("controller_repository"), CONTROLLER_REPOSITORY))
    if state.get("protocol_name") != PROTOCOL_NAME:
        raise CameraStateOwnershipError("Camera state protocol at %s is %r, not %s; refusing to operate on it."
                                        % (STATE_FILE, state.get("protocol_name"), PROTOCOL_NAME))
    return state


def build_state_from_args(args):
    if not getattr(args, "mouse_id", None):
        raise RuntimeError(
            "No saved camera session state found. Pass `--mouse-id` and `--session-id`, "
            "or run `start` first."
        )
    if not getattr(args, "session_id", None):
        raise RuntimeError(
            "No saved camera session state found. Pass `--session-id` as well, "
            "or run `start` first so the session ID is saved automatically."
        )

    paths = make_session_paths(args)
    camera_host = resolve_camera_host(args)
    return {
        "state_schema_version": CAMERA_STATE_SCHEMA_VERSION,
        "controller_repository": CONTROLLER_REPOSITORY,
        "protocol_name": PROTOCOL_NAME,
        "created_utc": utc_iso_now(),
        "camera_host": camera_host,
        "framerate": getattr(args, "framerate", CAMERA_FRAMERATE),
        "remote_camera_repo": getattr(args, "remote_camera_repo", REMOTE_CAMERA_REPO),
        "remote_camera_start": getattr(args, "remote_camera_start", REMOTE_CAMERA_START),
        "remote_camera_stop": getattr(args, "remote_camera_stop", REMOTE_CAMERA_STOP),
        **paths,
    }


def resolve_camera_host(args, state=None):
    if getattr(args, "camera_host", None):
        return args.camera_host
    if state and state.get("camera_host"):
        return state["camera_host"]
    return DEFAULT_CAMERA_HOST


def make_session_paths(args):
    mouse_id = sanitize_id(args.mouse_id)
    if not mouse_id:
        raise RuntimeError("mouse ID cannot be empty")

    session_id = sanitize_id(args.session_id) if args.session_id else make_session_name(mouse_id, utc_label())

    local_session_dir = (LOCAL_VIDEO_ROOT / session_id).resolve()
    local_video_dir = local_session_dir / "video"

    remote_session_dir = "%s/%s" % (REMOTE_VIDEO_ROOT, session_id)
    remote_video_dir = "%s/video" % remote_session_dir
    remote_base_path = "%s/%s" % (remote_video_dir, session_id)
    remote_log_file = "%s/camera_acquisition.log" % remote_video_dir
    remote_pid_file = "%s/camera_acquisition.pid" % remote_video_dir

    return {
        "mouse_id": mouse_id,
        "session_id": session_id,
        "local_session_dir": str(local_session_dir),
        "local_video_dir": str(local_video_dir),
        "remote_session_dir": remote_session_dir,
        "remote_video_dir": remote_video_dir,
        "remote_base_path": remote_base_path,
        "remote_log_file": remote_log_file,
        "remote_pid_file": remote_pid_file,
    }


def start_camera(args):
    camera_host = resolve_camera_host(args)
    paths = make_session_paths(args)
    local_video_dir = Path(paths["local_video_dir"])
    local_video_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "state_schema_version": CAMERA_STATE_SCHEMA_VERSION,
        "controller_repository": CONTROLLER_REPOSITORY,
        "protocol_name": PROTOCOL_NAME,
        "created_utc": utc_iso_now(),
        "camera_host": camera_host,
        "framerate": args.framerate,
        "remote_camera_repo": args.remote_camera_repo,
        "remote_camera_start": args.remote_camera_start,
        "remote_camera_stop": args.remote_camera_stop,
        "camera_status": "launch_requested",
        "camera_pid": None,
        "camera_start_command_returned": False,
        "camera_pid_confirmed": False,
        "camera_process_confirmed": False,
        "camera_output_file_detected": False,
        "camera_output_growing_confirmed": False,
        "camera_start_failed": False,
        "camera_start_requested_utc": utc_iso_now(),
        "camera_launch_returned_utc": None,
        "camera_output_growth_confirmed_utc": None,
        "camera_start_confirmed_utc": None,
        **paths,
    }

    append_event(local_video_dir, "camera_start_requested", state)
    save_state(state)

    remote_log = paths["remote_log_file"]
    remote_pid_file = paths["remote_pid_file"]
    launch_cmd = build_remote_camera_launch_command(
        paths,
        args.framerate,
        args.remote_camera_repo,
        args.remote_camera_start,
        remote_log,
        remote_pid_file,
    )

    launch_timed_out = False
    launch_stdout = ""
    try:
        launch_result = run_ssh(
            camera_host,
            launch_cmd,
            dry_run=args.dry_run,
            timeout=CAMERA_START_LAUNCH_TIMEOUT_SEC,
        )
        launch_stdout = launch_result.stdout or ""
        state["camera_start_command_returned"] = True
        state["camera_status"] = "launch_returned"
        state["camera_launch_returned_utc"] = utc_iso_now()
        state["camera_pid"] = parse_camera_pid(launch_stdout) or state.get("camera_pid")
        save_state(state)
        append_event(
            local_video_dir,
            "camera_start_returned",
            {
                **state,
                "launch_stdout": launch_stdout,
            },
        )
    except Exception as exc:
        if "timed out" in str(exc).lower():
            launch_timed_out = True
            state["camera_launch_timed_out"] = True
            print("Camera launch SSH timed out; checking whether the remote process still started.")
            save_state(state)
        else:
            state["camera_start_failed"] = True
            state["camera_status"] = "start_failed"
            state["camera_start_failed_utc"] = utc_iso_now()
            state["camera_start_error"] = str(exc)
            save_state(state)
            append_event(
                local_video_dir,
                "camera_start_failed",
                {
                    "camera_host": camera_host,
                    "error": str(exc),
                },
            )
            try:
                state["camera_cleanup_after_start_failure_attempted"] = True
                stop_camera(args, state)
                state["camera_cleanup_after_start_failure_confirmed"] = True
                save_state(state)
            except Exception as cleanup_exc:
                state["camera_cleanup_after_start_failure_confirmed"] = False
                state["camera_cleanup_after_start_failure_error"] = str(cleanup_exc)
                save_state(state)
                raise RuntimeError(
                    "Camera launch failed, and cleanup could not be confirmed: %s"
                    % cleanup_exc
                ) from exc
            raise RuntimeError("Camera launch failed; cleanup was confirmed.") from exc

    try:
        ready = wait_for_remote_camera_ready(
            camera_host,
            remote_pid_file,
            remote_log,
            dry_run=args.dry_run,
            timeout_sec=CAMERA_START_READY_TIMEOUT_SEC,
        )
        camera_pid = ready.get("camera_pid")
        if camera_pid is not None:
            state["camera_pid"] = camera_pid
        state["camera_status"] = "process_confirmed"
        state["camera_pid_confirmed"] = True
        state["camera_process_confirmed"] = True
        state["camera_process_confirmed_utc"] = utc_iso_now()
        save_state(state)
        append_event(
            local_video_dir,
            "camera_process_confirmed",
            {
                "camera_host": camera_host,
                "camera_pid": state.get("camera_pid"),
                "remote_pid_file": remote_pid_file,
            },
        )
        print("Camera acquisition process is running; waiting for video output...")

        output_ready = wait_for_remote_camera_output_growth(
            camera_host,
            remote_pid_file,
            paths["remote_video_dir"],
            remote_log,
            dry_run=args.dry_run,
            timeout_sec=CAMERA_OUTPUT_READY_TIMEOUT_SEC,
        )
        state["camera_status"] = "recording_confirmed"
        state["camera_output_file_detected"] = output_ready["camera_output_file_detected"]
        state["camera_output_growing_confirmed"] = output_ready["camera_output_growing_confirmed"]
        state["camera_output_file"] = output_ready["camera_output_file"]
        state["camera_output_initial_size_bytes"] = output_ready["initial_size_bytes"]
        state["camera_output_confirmed_size_bytes"] = output_ready["confirmed_size_bytes"]
        state["camera_readiness_reference"] = "pid_alive_and_output_size_increasing"
        state["camera_output_growth_confirmed_utc"] = utc_iso_now()
        state["camera_start_confirmed_utc"] = state["camera_output_growth_confirmed_utc"]
        state["camera_launch_timed_out"] = launch_timed_out
        save_state(state)
        append_event(
            local_video_dir,
            "camera_recording_confirmed",
            {
                "camera_host": camera_host,
                "camera_pid": state.get("camera_pid"),
                "camera_pid_confirmed": True,
                "camera_process_confirmed": True,
                "camera_output_file_detected": True,
                "camera_output_growing_confirmed": True,
                "camera_output_file": state.get("camera_output_file"),
                "camera_readiness_reference": state["camera_readiness_reference"],
                "camera_launch_timed_out": launch_timed_out,
                "remote_pid_file": remote_pid_file,
                "remote_log_file": remote_log,
            },
        )
    except Exception as exc:
        tail = tail_remote_log(camera_host, remote_log, dry_run=args.dry_run)
        print("Camera startup failed. Remote log tail\n%s" % (tail or "<no log output>"), file=sys.stderr)
        state["camera_start_failed"] = True
        state["camera_status"] = "start_failed"
        state["camera_start_failed_utc"] = utc_iso_now()
        state["camera_start_error"] = str(exc)
        save_state(state)
        append_event(
            local_video_dir,
            "camera_start_failed",
            {
                "camera_host": camera_host,
                "error": str(exc),
                "remote_log_tail": tail,
            },
        )
        try:
            state["camera_cleanup_after_start_failure_attempted"] = True
            stop_camera(args, state)
            state["camera_cleanup_after_start_failure_confirmed"] = True
            save_state(state)
        except Exception as cleanup_exc:
            state["camera_cleanup_after_start_failure_confirmed"] = False
            state["camera_cleanup_after_start_failure_error"] = str(cleanup_exc)
            save_state(state)
            raise RuntimeError(
                "Camera recording did not become ready, and cleanup could not be confirmed: %s"
                % cleanup_exc
            ) from exc
        raise RuntimeError("Camera recording did not become ready; cleanup was confirmed.") from exc

    print("Camera video output confirmed growing.")
    print("Camera host:      %s" % camera_host)
    print("Remote video dir: %s" % paths["remote_video_dir"])
    print("Local video dir:  %s" % local_video_dir)
    return state


def stop_camera(args, state=None):
    if state is None:
        state = load_state()

    camera_host = resolve_camera_host(args, state)
    local_video_dir = Path(state.get("local_video_dir", LOCAL_VIDEO_ROOT / "unknown" / "video"))
    local_video_dir.mkdir(parents=True, exist_ok=True)
    remote_pid_file = state.get("remote_pid_file") or "%s/camera_acquisition.pid" % state.get("remote_video_dir", REMOTE_VIDEO_ROOT)
    remote_video_dir = state.get("remote_video_dir") or REMOTE_VIDEO_ROOT
    remote_base_path = state.get("remote_base_path") or remote_video_dir
    remote_log_file = state.get("remote_log_file") or "%s/camera_acquisition.log" % remote_video_dir
    expected_pid = state.get("camera_pid")
    ignore_stop_errors = getattr(args, "ignore_stop_errors", False)

    append_event(local_video_dir, "camera_stop_requested", {"camera_host": camera_host, "remote_pid_file": remote_pid_file})
    state["camera_status"] = "stop_requested"
    state["camera_stop_requested_utc"] = utc_iso_now()
    save_state(state)

    remote_stop = getattr(args, "remote_camera_stop", None) or state.get("remote_camera_stop") or REMOTE_CAMERA_STOP
    stop_cmd = build_remote_camera_stop_command(remote_pid_file, remote_stop)
    run_ssh(
        camera_host,
        stop_cmd,
        check=not ignore_stop_errors,
        dry_run=args.dry_run,
        timeout=CAMERA_STOP_TIMEOUT_SEC,
    )

    state["camera_stop_command_returned"] = True
    state["camera_stop_command_returned_utc"] = utc_iso_now()
    state["camera_stop_returned_utc"] = state["camera_stop_command_returned_utc"]
    save_state(state)
    append_event(
        local_video_dir,
        "camera_stop_command_returned",
        {"camera_host": camera_host, "remote_pid_file": remote_pid_file},
    )

    try:
        wait_for_remote_camera_stopped(
            camera_host,
            expected_pid,
            remote_base_path,
            remote_log_file,
            dry_run=args.dry_run,
            timeout_sec=CAMERA_STOP_VERIFY_TIMEOUT_SEC,
        )
    except Exception as exc:
        state["camera_status"] = "stop_unverified"
        state["camera_stop_confirmed"] = False
        state["camera_stop_verification_error"] = str(exc)
        save_state(state)
        append_event(
            local_video_dir,
            "camera_stop_unverified",
            {
                "camera_host": camera_host,
                "remote_pid_file": remote_pid_file,
                "remote_base_path": remote_base_path,
                "error": str(exc),
            },
        )
        if ignore_stop_errors:
            print(
                "WARNING: Camera stop command returned, but stopped state was not verified.",
                file=sys.stderr,
            )
            return state
        raise

    run_ssh(
        camera_host,
        "rm -f %s" % shlex.quote(remote_pid_file),
        dry_run=args.dry_run,
        timeout=CAMERA_STOP_TIMEOUT_SEC,
    )
    state["camera_status"] = "stopped_confirmed"
    state["camera_stop_confirmed"] = True
    state["camera_stop_confirmed_utc"] = utc_iso_now()
    save_state(state)
    append_event(
        local_video_dir,
        "camera_stop_confirmed",
        {
            "camera_host": camera_host,
            "remote_pid_file": remote_pid_file,
            "remote_base_path": remote_base_path,
            "expected_pid": expected_pid,
        },
    )
    print("Camera stop verified.")
    return state


def preview_camera(args):
    camera_host = resolve_camera_host(args)
    preview_cmd = (
        "set -e; "
        "cam=$(command -v rpicam-hello || command -v libcamera-hello); "
        "if [ -z \"$cam\" ]; then echo 'No rpicam-hello or libcamera-hello found' >&2; exit 1; fi; "
        "mkdir -p /home/pi/stim_logs; "
        "nohup \"$cam\" -t 0 --fullscreen </dev/null >%s 2>&1 & echo $! > %s"
        % (shlex.quote(REMOTE_CAMERA_PREVIEW_LOG), shlex.quote(REMOTE_CAMERA_PREVIEW_PID_FILE))
    )
    stop_cmd = (
        "if [ -f %s ]; then "
        "pid=$(cat %s); "
        "kill \"$pid\" 2>/dev/null || true; "
        "sleep 0.5; "
        "kill -9 \"$pid\" 2>/dev/null || true; "
        "rm -f %s; "
        "fi"
        % (
            shlex.quote(REMOTE_CAMERA_PREVIEW_PID_FILE),
            shlex.quote(REMOTE_CAMERA_PREVIEW_PID_FILE),
            shlex.quote(REMOTE_CAMERA_PREVIEW_PID_FILE),
        )
    )

    print("Starting remote camera preview on %s..." % camera_host)
    run_ssh(camera_host, preview_cmd, dry_run=args.dry_run, timeout=CAMERA_PREVIEW_LAUNCH_TIMEOUT_SEC)
    print("Preview started. Type y and Enter to stop it.")
    if not args.dry_run:
        while True:
            try:
                response = input("> ").strip().lower()
            except EOFError:
                response = "y"
            if response == "y":
                break
            print("Preview still running. Type y and Enter to stop.")
        run_ssh(camera_host, stop_cmd, dry_run=args.dry_run, timeout=CAMERA_PREVIEW_STOP_TIMEOUT_SEC)
        print("Preview stopped.")
    else:
        print("Dry run finished; preview was not actually started.")


def convert_camera(args, state=None):
    if getattr(args, "ffmpeg_timeout_sec", CAMERA_FFMPEG_TIMEOUT_SEC) <= 0:
        raise ValueError("--ffmpeg-timeout-sec must be greater than 0")

    if state is None:
        try:
            state = load_state()
        except CameraStateNotFoundError:
            state = build_state_from_args(args)

    camera_host = resolve_camera_host(args, state)
    local_video_dir = Path(state["local_video_dir"])
    local_video_dir.mkdir(parents=True, exist_ok=True)
    manifest = state.get("remote_raw_manifest") or []
    if manifest:
        try:
            verify_local_camera_raw_manifest(local_video_dir, manifest)
            state["camera_raw_files_verified"] = True
            state["camera_raw_hash_verified"] = True
        except Exception as exc:
            state["camera_raw_files_verified"] = False
            state["camera_raw_hash_verified"] = False
            state["camera_conversion_completed"] = False
            state["camera_mp4_verified"] = False
            state["camera_conversion_failed"] = True
            state["remote_raw_retained"] = True
            state["camera_conversion_error"] = "Current local raw verification failed: %s" % exc
            save_state(state)
            return state

    append_event(
        local_video_dir,
        "camera_conversion_requested",
        {
            "camera_host": camera_host,
            "local_video_dir": str(local_video_dir),
            "ffmpeg_timeout_sec": getattr(args, "ffmpeg_timeout_sec", CAMERA_FFMPEG_TIMEOUT_SEC),
        },
    )

    try:
        conversion_state = run_camera_conversion_workflow(
            camera_host,
            local_video_dir,
            framerate=state.get("framerate", CAMERA_FRAMERATE),
            dry_run=args.dry_run,
            timeout_sec=getattr(args, "ffmpeg_timeout_sec", CAMERA_FFMPEG_TIMEOUT_SEC),
        )
    except Exception as exc:
        state["camera_conversion_failed"] = True
        state["camera_conversion_failed_utc"] = utc_iso_now()
        state["camera_conversion_error"] = str(exc)
        state["camera_conversion_completed"] = False
        state["camera_mp4_verified"] = False
        state["camera_conversion_deferred"] = False
        append_event(
            local_video_dir,
            "camera_conversion_failed",
            {
                "camera_host": camera_host,
                "local_video_dir": str(local_video_dir),
                "ffmpeg_timeout_sec": getattr(args, "ffmpeg_timeout_sec", CAMERA_FFMPEG_TIMEOUT_SEC),
                "error": str(exc),
            },
        )
        print("MP4 conversion did not complete.", file=sys.stderr)
        return state

    state.update(conversion_state)
    state["camera_conversion_completed_utc"] = utc_iso_now() if conversion_state.get("camera_conversion_completed") else ""
    state["camera_conversion_failed"] = bool(conversion_state.get("camera_conversion_error"))
    state["camera_conversion_failed_utc"] = utc_iso_now() if state["camera_conversion_failed"] else ""
    state["camera_mp4_verified"] = bool(conversion_state.get("camera_conversion_completed"))
    maybe_cleanup_remote_raw(args, state, camera_host, local_video_dir)
    save_state(state)
    if conversion_state.get("camera_conversion_completed"):
        print("Converted camera files in: %s" % local_video_dir)
    return state

def fetch_camera(args, state=None):
    if getattr(args, "rsync_timeout_sec", CAMERA_RSYNC_TIMEOUT_SEC) <= 0:
        raise ValueError("--rsync-timeout-sec must be greater than 0")
    if getattr(args, "ffmpeg_timeout_sec", CAMERA_FFMPEG_TIMEOUT_SEC) <= 0:
        raise ValueError("--ffmpeg-timeout-sec must be greater than 0")

    if state is None:
        try:
            state = load_state()
        except CameraStateNotFoundError:
            state = build_state_from_args(args)

    camera_host = resolve_camera_host(args, state)
    remote_video_dir = state["remote_video_dir"]
    local_video_dir = Path(state["local_video_dir"])
    local_video_dir.mkdir(parents=True, exist_ok=True)
    if not state.get("camera_stop_confirmed"):
        raise RuntimeError("Refusing to fetch/clean camera data before camera stop is confirmed.")
    if args.dry_run:
        state.update({"camera_transfer_completed": False, "camera_fetch_completed": False,
                      "remote_raw_retained": True, "remote_raw_cleanup_completed": False,
                      "camera_fetch_status": "dry_run"})
        append_event(local_video_dir, "camera_fetch_dry_run", {"remote_video_dir": remote_video_dir})
        return state
    if state.get("remote_raw_cleanup_completed") and state.get("remote_raw_manifest"):
        try:
            verify_local_camera_raw_manifest(local_video_dir, state["remote_raw_manifest"])
            for item in state["remote_raw_manifest"]:
                probe_mp4(local_video_dir / Path(item["filename"]).with_suffix(".mp4"))
        except Exception as exc:
            state.update({"camera_fetch_completed": False, "camera_fetch_status": "degraded_after_cleanup",
                          "remote_raw_retained": False, "camera_fetch_error": str(exc)})
            save_state(state)
            return state
        state.update({"camera_transfer_completed": True, "camera_fetch_completed": True,
                      "camera_raw_files_verified": True, "camera_raw_hash_verified": True,
                      "camera_mp4_verified": True, "camera_fetch_status": "already_secured",
                      "remote_raw_retained": False})
        save_state(state)
        return state
    state["camera_transfer_command_completed"] = False
    state["camera_fetch_completed"] = False
    state["camera_conversion_attempted"] = False
    state["camera_conversion_completed"] = False
    state["camera_conversion_deferred"] = False
    state["camera_fetch_timed_out"] = False
    state["camera_raw_files_verified"] = False
    state["camera_raw_hash_verified"] = False
    state["camera_raw_files_verified"] = False
    state["raw_h264_available_locally"] = False
    state["remote_raw_retained"] = True
    state["remote_raw_cleanup_attempted"] = False
    state["remote_raw_cleanup_completed"] = False
    state["remote_raw_cleanup_error"] = ""
    state["converted_mp4_files"] = []

    append_event(
        local_video_dir,
        "camera_fetch_requested",
        {
            "camera_host": camera_host,
            "remote_video_dir": remote_video_dir,
            "local_video_dir": str(local_video_dir),
            "rsync_timeout_sec": getattr(args, "rsync_timeout_sec", CAMERA_RSYNC_TIMEOUT_SEC),
            "ffmpeg_timeout_sec": getattr(args, "ffmpeg_timeout_sec", CAMERA_FFMPEG_TIMEOUT_SEC),
        },
    )

    try:
        manifest = collect_remote_raw_manifest(camera_host, remote_video_dir, dry_run=args.dry_run)
        state["remote_raw_manifest"] = manifest
        state["camera_raw_file_count"] = len(manifest)
        write_video_manifest(local_video_dir, manifest)
        save_state(state)
        run_rsync(
            camera_host,
            remote_video_dir,
            local_video_dir,
            dry_run=args.dry_run,
            timeout_sec=getattr(args, "rsync_timeout_sec", CAMERA_RSYNC_TIMEOUT_SEC),
        )
    except Exception as exc:
        state["camera_fetch_failed"] = True
        state["camera_fetch_failed_utc"] = utc_iso_now()
        state["camera_fetch_error"] = str(exc)
        if "timed out" in str(exc).lower():
            state["camera_fetch_timed_out"] = True
        append_event(
            local_video_dir,
            "camera_fetch_failed",
            {
                "camera_host": camera_host,
                "remote_video_dir": remote_video_dir,
                "local_video_dir": str(local_video_dir),
                "stage": "rsync",
                "timeout_sec": getattr(args, "rsync_timeout_sec", CAMERA_RSYNC_TIMEOUT_SEC),
                "error": str(exc),
            },
        )
        save_state(state)
        raise

    state["camera_transfer_command_completed"] = True
    state["camera_fetch_returned_utc"] = utc_iso_now()
    try:
        verified_raw_files = verify_local_camera_raw_manifest(local_video_dir, state.get("remote_raw_manifest", []))
    except Exception as exc:
        state["camera_fetch_failed"] = True
        state["camera_fetch_error"] = str(exc)
        state["remote_raw_retained"] = True
        append_event(local_video_dir, "camera_raw_manifest_verification_failed", {"error": str(exc)})
        save_state(state)
        raise
    state["camera_raw_files_verified"] = True
    state["camera_raw_hash_verified"] = True
    state["raw_h264_available_locally"] = bool(verified_raw_files)
    append_event(
        local_video_dir,
        "camera_fetch_returned",
        {
            "camera_host": camera_host,
            "remote_video_dir": remote_video_dir,
            "local_video_dir": str(local_video_dir),
        },
    )

    append_event(
        local_video_dir,
        "camera_conversion_requested",
        {
            "camera_host": camera_host,
            "local_video_dir": str(local_video_dir),
            "ffmpeg_timeout_sec": getattr(args, "ffmpeg_timeout_sec", CAMERA_FFMPEG_TIMEOUT_SEC),
        },
    )

    try:
        conversion_state = run_camera_conversion_workflow(
            camera_host,
            local_video_dir,
            framerate=state.get("framerate", CAMERA_FRAMERATE),
            dry_run=args.dry_run,
            timeout_sec=getattr(args, "ffmpeg_timeout_sec", CAMERA_FFMPEG_TIMEOUT_SEC),
        )
    except Exception as exc:
        state["camera_fetch_failed"] = True
        state["camera_fetch_failed_utc"] = utc_iso_now()
        state["camera_fetch_error"] = str(exc)
        append_event(
            local_video_dir,
            "camera_fetch_failed",
            {
                "camera_host": camera_host,
                "remote_video_dir": remote_video_dir,
                "local_video_dir": str(local_video_dir),
                "stage": "raw_verification",
                "error": str(exc),
            },
        )
        save_state(state)
        raise

    state.update(conversion_state)
    state["camera_fetch_completed"] = True
    state["camera_fetch_completed_utc"] = utc_iso_now()
    state["raw_h264_available_locally"] = bool(conversion_state.get("raw_h264_available_locally"))
    state["camera_raw_files_verified"] = bool(conversion_state.get("camera_raw_files_verified"))
    state["camera_raw_file_count"] = conversion_state.get("camera_raw_file_count", 0)
    state["camera_conversion_attempted"] = bool(conversion_state.get("camera_conversion_attempted"))
    state["camera_conversion_completed"] = bool(conversion_state.get("camera_conversion_completed"))
    state["camera_conversion_deferred"] = bool(conversion_state.get("camera_conversion_deferred"))
    state["camera_conversion_error"] = conversion_state.get("camera_conversion_error", "")
    state["camera_conversion_skip_reason"] = conversion_state.get("camera_conversion_skip_reason", "")
    state["converted_mp4_files"] = conversion_state.get("converted_mp4_files", [])
    state["camera_mp4_verified"] = bool(conversion_state.get("camera_conversion_completed"))
    maybe_cleanup_remote_raw(args, state, camera_host, local_video_dir)
    save_state(state)
    print("Fetched camera files to: %s" % local_video_dir)
    return state

def status_camera(args):
    state = load_state() if STATE_FILE.exists() else None
    camera_host = resolve_camera_host(args, state)
    safe_start_pattern = "[v]ideo_acquisition/start_acquisition.py"
    remote_cmd = (
        "echo '--- camera acquisition processes ---'; "
        "pgrep -af %s || true; "
        "echo '--- recent camera logs ---'; "
        "find /home/pi/stim_logs -name 'camera_acquisition.log' -type f 2>/dev/null | tail -n 5 || true"
        % shlex.quote(safe_start_pattern)
    )
    run_ssh(camera_host, remote_cmd, dry_run=args.dry_run)


def print_last_state(args):
    print(json.dumps(load_state(), indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="Standalone second-Pi camera controller for vstim_natural_img_reward.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--camera-host",
        default=None,
        help="SSH host for camera Pi. Default: %s" % DEFAULT_CAMERA_HOST,
    )
    common.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    common.add_argument("--json", action="store_true", help="Also print a machine-readable result line.")

    start = sub.add_parser("start", parents=[common], help="Start remote camera recording.")
    start.add_argument("--mouse-id", required=True, help="Mouse ID for session folder.")
    start.add_argument("--session-id", default=None, help="Optional session ID. Default: mouse_UTCtimestamp.")
    start.add_argument("--framerate", type=int, default=CAMERA_FRAMERATE)
    start.add_argument("--remote-camera-repo", default=REMOTE_CAMERA_REPO)
    start.add_argument("--remote-camera-start", default=REMOTE_CAMERA_START)
    start.add_argument("--remote-camera-stop", default=REMOTE_CAMERA_STOP)
    start.set_defaults(func=start_camera)

    stop = sub.add_parser("stop", parents=[common], help="Stop remote camera recording.")
    stop.add_argument("--remote-camera-stop", default=REMOTE_CAMERA_STOP)
    stop.add_argument("--ignore-stop-errors", action="store_true", default=False)
    stop.set_defaults(func=stop_camera)

    fetch = sub.add_parser("fetch", parents=[common], help="Fetch last remote camera files with rsync.")
    fetch.add_argument("--rsync-timeout-sec", type=float, default=CAMERA_RSYNC_TIMEOUT_SEC)
    fetch.add_argument("--ffmpeg-timeout-sec", type=float, default=CAMERA_FFMPEG_TIMEOUT_SEC)
    fetch.set_defaults(func=fetch_camera)

    convert = sub.add_parser("convert", parents=[common], help="Convert locally fetched .h264 files to MP4.")
    convert.add_argument("--ffmpeg-timeout-sec", type=float, default=CAMERA_FFMPEG_TIMEOUT_SEC)
    convert.set_defaults(func=convert_camera)

    preview = sub.add_parser("preview", parents=[common], help="Start a live camera preview, then stop it when you type y.")
    preview.set_defaults(func=preview_camera)

    stop_fetch = sub.add_parser("stop-fetch", parents=[common], help="Stop recording, then fetch files.")
    stop_fetch.add_argument("--mouse-id", default=None, help="Mouse ID if no saved session state exists yet.")
    stop_fetch.add_argument("--session-id", default=None, help="Session ID if no saved session state exists yet.")
    stop_fetch.add_argument("--remote-camera-stop", default=REMOTE_CAMERA_STOP)
    stop_fetch.add_argument("--ignore-stop-errors", action="store_true", default=False)
    stop_fetch.add_argument("--rsync-timeout-sec", type=float, default=CAMERA_RSYNC_TIMEOUT_SEC)
    stop_fetch.add_argument("--ffmpeg-timeout-sec", type=float, default=CAMERA_FFMPEG_TIMEOUT_SEC)

    def do_stop_fetch(args):
        try:
            state = load_state()
        except CameraStateNotFoundError:
            state = build_state_from_args(args)
        state = stop_camera(args, state)
        time.sleep(2.0)
        return fetch_camera(args, state)

    stop_fetch.set_defaults(func=do_stop_fetch)

    status = sub.add_parser("status", parents=[common], help="Check whether camera acquisition is running.")
    status.set_defaults(func=status_camera)

    last = sub.add_parser("last-state", help="Print last camera session state.")
    last.set_defaults(func=print_last_state)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    result = args.func(args)
    if getattr(args, "json", False) and result is not None:
        print(JSON_RESULT_PREFIX + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise
