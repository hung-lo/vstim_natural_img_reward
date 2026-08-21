#!/usr/bin/env python3
"""Hardware-free safety tests for the reward-conditioning camera controller."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import remote_camera_control as camera


class CameraControlSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="camera_control_test_")
        self.addCleanup(self.temp_dir.cleanup)
        self.state_file = Path(self.temp_dir.name) / "camera_state.json"
        self.state_patch = mock.patch.object(camera, "STATE_FILE", self.state_file)
        self.state_patch.start()
        self.addCleanup(self.state_patch.stop)

    def _state(self):
        return {
            "controller_repository": camera.CONTROLLER_REPOSITORY,
            "protocol_name": camera.PROTOCOL_NAME,
            "session_id": "mouse_session",
            "mouse_id": "mouse",
            "camera_host": "pi@test",
            "remote_video_dir": "/remote/mouse_session/video",
            "remote_base_path": "/remote/mouse_session/video/mouse_session",
            "local_video_dir": str(Path(self.temp_dir.name) / "video"),
        }

    def test_reward_specific_state_and_atomic_save(self):
        camera.save_state(self._state())
        self.assertEqual(camera.load_state()["controller_repository"], camera.CONTROLLER_REPOSITORY)
        self.assertEqual(json.loads(self.state_file.read_text())["protocol_name"], camera.PROTOCOL_NAME)

    def test_rejects_foreign_state(self):
        state = self._state()
        state["controller_repository"] = "vstim_natural"
        self.state_file.write_text(json.dumps(state))
        with self.assertRaisesRegex(RuntimeError, "refusing"):
            camera.load_state()

    def test_rsync_is_non_destructive(self):
        with mock.patch.object(camera, "run_cmd") as run_cmd:
            run_cmd.return_value = mock.Mock()
            camera.run_rsync("pi@test", "/remote/video", Path(self.temp_dir.name), dry_run=True)
        command = run_cmd.call_args.args[0]
        self.assertNotIn("--remove-source-files", command)
        self.assertIn("--append-verify", command)

    def test_manifest_verification_rejects_wrong_size_or_hash(self):
        video_dir = Path(self.temp_dir.name) / "video"
        video_dir.mkdir()
        raw = video_dir / "video.h264"
        raw.write_bytes(b"camera bytes")
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        manifest = [{"remote_path": "/remote/video/video.h264", "filename": "video.h264", "size_bytes": raw.stat().st_size, "sha256": digest}]
        self.assertEqual(camera.verify_local_camera_raw_manifest(video_dir, manifest), [raw])
        manifest[0]["size_bytes"] += 1
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            camera.verify_local_camera_raw_manifest(video_dir, manifest)
        manifest[0]["size_bytes"] -= 1
        manifest[0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            camera.verify_local_camera_raw_manifest(video_dir, manifest)

    def test_cleanup_is_limited_to_manifest_paths(self):
        command = camera.build_remote_cleanup_command(
            [{"remote_path": "/remote/session/video/a.h264"}], "/remote/session/video"
        )
        self.assertIn("/remote/session/video/a.h264", command)
        self.assertNotIn("*", command)
        with self.assertRaises(RuntimeError):
            camera.build_remote_cleanup_command(
                [{"remote_path": "/remote/other/a.h264"}], "/remote/session/video"
            )

    def test_mp4_probe_validation(self):
        valid = {"streams": [{"codec_name": "h264", "width": 640, "height": 480, "avg_frame_rate": "30/1"}], "format": {"duration": "1.5"}}
        with mock.patch.object(camera.shutil, "which", return_value="ffprobe"), mock.patch.object(camera, "run_cmd", return_value=mock.Mock(stdout=json.dumps(valid))):
            self.assertEqual(camera.probe_mp4("video.mp4")["width"], 640)
        valid["format"]["duration"] = "0"
        with mock.patch.object(camera.shutil, "which", return_value="ffprobe"), mock.patch.object(camera, "run_cmd", return_value=mock.Mock(stdout=json.dumps(valid))):
            with self.assertRaises(RuntimeError):
                camera.probe_mp4("video.mp4")


if __name__ == "__main__":
    unittest.main()
