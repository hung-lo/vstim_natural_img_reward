#!/usr/bin/env python3
"""Unit tests for best-effort monitor telemetry."""

import json
import queue
import socket
import time
import unittest
from unittest import mock

import rig_telemetry


class _LiveProcess:
    def is_alive(self):
        return True


class _FullQueue:
    def put_nowait(self, value):
        raise queue.Full


class RigTelemetryTests(unittest.TestCase):
    def test_disabled_publisher_is_safe_noop(self):
        publisher = rig_telemetry.TelemetryPublisher(enabled=False)
        self.assertFalse(publisher.start())
        self.assertFalse(publisher.publish_state({"phase": "PRE"}))
        publisher.close()

    def test_queue_full_publish_returns_without_raising(self):
        publisher = rig_telemetry.TelemetryPublisher(enabled=True)
        publisher.message_queue = _FullQueue()
        publisher.process = _LiveProcess()
        started = time.monotonic()
        self.assertFalse(publisher.publish_state({"phase": "PRE"}))
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertEqual(publisher.dropped_count, 1)

    def test_udp_child_adds_common_envelope(self):
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            receiver.bind(("127.0.0.1", 0))
        except PermissionError:
            receiver.close()
            self.skipTest("sandbox does not permit localhost UDP sockets")
        receiver.settimeout(1.0)
        publisher = rig_telemetry.TelemetryPublisher(
            host="127.0.0.1", port=receiver.getsockname()[1],
            session_id="session123", join_timeout_sec=0.5,
        )
        try:
            self.assertTrue(publisher.start())
            self.assertTrue(publisher.publish_state({"phase": "PRE", "trial": 1}))
            packet, _ = receiver.recvfrom(4096)
            decoded = json.loads(packet.decode("utf-8"))
            self.assertEqual(decoded["schema_version"], 1)
            self.assertEqual(decoded["type"], "state")
            self.assertEqual(decoded["protocol"], "reward_conditioning")
            self.assertEqual(decoded["session_id"], "session123")
            self.assertTrue(decoded["timestamp_utc"].endswith("Z"))
            self.assertEqual(decoded["phase"], "PRE")
        finally:
            publisher.close()
            receiver.close()

    def test_worker_send_failure_does_not_escape(self):
        message_queue = mock.Mock()
        message_queue.get.side_effect = [
            {"type": "state", "payload": {"phase": "PRE"}}, None
        ]
        with mock.patch.object(rig_telemetry.socket, "socket") as make_socket:
            sock = make_socket.return_value
            sock.sendto.side_effect = OSError("unreachable")
            rig_telemetry._telemetry_worker(
                message_queue, "127.0.0.1", 5055, "session", "reward_conditioning"
            )
            sock.close.assert_called_once_with()

    def test_close_terminates_stalled_worker_bounded(self):
        publisher = rig_telemetry.TelemetryPublisher(enabled=True, join_timeout_sec=0.01)
        process = mock.Mock()
        process.is_alive.side_effect = [True, True]
        publisher.message_queue = _FullQueue()
        publisher.process = process
        started = time.monotonic()
        publisher.close()
        self.assertLess(time.monotonic() - started, 0.2)
        process.terminate.assert_called_once_with()


if __name__ == "__main__":
    unittest.main(verbosity=2)
