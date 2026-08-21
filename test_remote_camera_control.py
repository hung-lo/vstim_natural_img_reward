#!/usr/bin/env python3
"""Hardware-free safety tests for the reward-conditioning camera controller."""

import hashlib
import json
import tempfile
import types
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
            "state_schema_version": camera.CAMERA_STATE_SCHEMA_VERSION,
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

    def test_manifest_uses_long_hash_timeout(self):
        with mock.patch.object(camera, "run_ssh", return_value=mock.Mock(stdout="/remote/video/a.h264\t1\t" + "a" * 64)) as run_ssh:
            camera.collect_remote_raw_manifest("pi@test", "/remote/video")
        self.assertEqual(run_ssh.call_args.kwargs["timeout"], camera.CAMERA_MANIFEST_TIMEOUT_SEC)

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

    def test_hash_mismatch_blocks_remote_cleanup(self):
        video_dir = Path(self.temp_dir.name) / "video"
        video_dir.mkdir()
        (video_dir / "a.h264").write_bytes(b"changed")
        state = self._state()
        state.update({"camera_stop_confirmed": True, "camera_raw_hash_verified": True,
                      "camera_mp4_verified": True, "remote_raw_manifest": [
                          {"remote_path": "/remote/mouse_session/video/a.h264", "filename": "a.h264", "size_bytes": 5, "sha256": "0" * 64}
                      ]})
        args = mock.Mock(dry_run=False)
        with mock.patch.object(camera, "run_ssh") as run_ssh:
            camera.maybe_cleanup_remote_raw(args, state, "pi@test", video_dir)
        run_ssh.assert_not_called()
        self.assertFalse(state["remote_raw_cleanup_completed"])
        self.assertTrue(state["remote_raw_retained"])
        self.assertFalse(state["camera_raw_files_verified"])
        self.assertFalse(state["camera_raw_hash_verified"])
        self.assertFalse(state["camera_mp4_verified"])

    def test_mp4_failure_blocks_cleanup_after_current_raw_verification(self):
        video_dir = Path(self.temp_dir.name) / "video"
        video_dir.mkdir()
        raw = video_dir / "a.h264"
        raw.write_bytes(b"camera")
        state = self._state()
        state.update({"camera_stop_confirmed": True, "remote_raw_manifest": [{
            "remote_path": "/remote/mouse_session/video/a.h264", "filename": "a.h264",
            "size_bytes": raw.stat().st_size, "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        }]})
        with mock.patch.object(camera, "probe_mp4", side_effect=RuntimeError("bad mp4")), mock.patch.object(camera, "run_ssh") as run_ssh:
            camera.maybe_cleanup_remote_raw(mock.Mock(dry_run=False), state, "pi@test", video_dir)
        run_ssh.assert_not_called()
        self.assertTrue(state["camera_raw_files_verified"])
        self.assertTrue(state["camera_raw_hash_verified"])
        self.assertFalse(state["camera_mp4_verified"])
        self.assertTrue(state["remote_raw_retained"])

    def test_fetch_dry_run_requires_no_remote_manifest_or_rsync(self):
        state = self._state()
        state["camera_stop_confirmed"] = True
        args = mock.Mock(dry_run=True, rsync_timeout_sec=1, ffmpeg_timeout_sec=1)
        with mock.patch.object(camera, "collect_remote_raw_manifest") as manifest, mock.patch.object(camera, "run_rsync") as rsync:
            result = camera.fetch_camera(args, state=state)
        manifest.assert_not_called()
        rsync.assert_not_called()
        self.assertEqual(result["camera_fetch_status"], "dry_run")

    def test_standalone_convert_clears_stale_success_after_raw_mismatch(self):
        video_dir = Path(self.temp_dir.name) / "video"
        video_dir.mkdir()
        (video_dir / "a.h264").write_bytes(b"changed")
        state = self._state()
        state.update({"camera_stop_confirmed": True, "camera_raw_files_verified": True,
                      "camera_raw_hash_verified": True, "camera_conversion_completed": True,
                      "camera_mp4_verified": True, "remote_raw_manifest": [{
                          "remote_path": "/remote/mouse_session/video/a.h264", "filename": "a.h264",
                          "size_bytes": 7, "sha256": "0" * 64,
                      }]})
        args = types.SimpleNamespace(dry_run=False, ffmpeg_timeout_sec=1, camera_host=None)
        with mock.patch.object(camera, "run_ssh") as run_ssh:
            result = camera.convert_camera(args, state=state)
        run_ssh.assert_not_called()
        self.assertFalse(result["camera_raw_hash_verified"])
        self.assertFalse(result["camera_conversion_completed"])
        self.assertFalse(result["camera_mp4_verified"])
        self.assertTrue(result["camera_conversion_failed"])

    def test_second_fetch_after_cleanup_is_already_secured(self):
        video_dir = Path(self.temp_dir.name) / "video"
        video_dir.mkdir()
        raw = video_dir / "a.h264"
        raw.write_bytes(b"camera")
        state = self._state()
        state.update({"camera_stop_confirmed": True, "remote_raw_cleanup_completed": True,
                      "remote_raw_retained": False, "remote_raw_manifest": [{
                          "remote_path": "/remote/mouse_session/video/a.h264", "filename": "a.h264",
                          "size_bytes": raw.stat().st_size, "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                      }]})
        args = types.SimpleNamespace(dry_run=False, rsync_timeout_sec=1, ffmpeg_timeout_sec=1, camera_host=None)
        with mock.patch.object(camera, "probe_mp4", return_value={"duration_sec": 1}), mock.patch.object(camera, "collect_remote_raw_manifest") as manifest, mock.patch.object(camera, "run_rsync") as rsync, mock.patch.object(camera, "run_ssh") as ssh:
            result = camera.fetch_camera(args, state=state)
        self.assertEqual(result["camera_fetch_status"], "already_secured")
        manifest.assert_not_called()
        rsync.assert_not_called()
        ssh.assert_not_called()

    def test_invalid_state_never_falls_back_for_recovery_commands(self):
        foreign = self._state()
        foreign["controller_repository"] = "vstim_natural"
        self.state_file.write_text(json.dumps(foreign))
        args = types.SimpleNamespace(dry_run=False, rsync_timeout_sec=1, ffmpeg_timeout_sec=1,
                                     camera_host=None, mouse_id="mouse", session_id="session")
        for command in (camera.fetch_camera, camera.convert_camera):
            with self.subTest(command=command.__name__), mock.patch.object(camera, "build_state_from_args") as fallback, mock.patch.object(camera, "run_ssh") as ssh:
                with self.assertRaises(camera.CameraStateOwnershipError):
                    command(args)
                fallback.assert_not_called()
                ssh.assert_not_called()

    def test_corrupt_and_schema_state_never_fall_back(self):
        args = types.SimpleNamespace(dry_run=False, rsync_timeout_sec=1, ffmpeg_timeout_sec=1,
                                     camera_host=None, mouse_id="mouse", session_id="session")
        self.state_file.write_text("not json")
        with mock.patch.object(camera, "build_state_from_args") as fallback:
            with self.assertRaises(camera.CameraStateCorruptError):
                camera.fetch_camera(args)
            fallback.assert_not_called()
        invalid = self._state()
        invalid["state_schema_version"] = 999
        self.state_file.write_text(json.dumps(invalid))
        with mock.patch.object(camera, "build_state_from_args") as fallback:
            with self.assertRaises(camera.CameraStateSchemaError):
                camera.convert_camera(args)
            fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
