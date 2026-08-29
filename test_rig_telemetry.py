#!/usr/bin/env python3
"""Unit tests for best-effort monitor telemetry."""

import json
import pickle
import socket
import time
import unittest
from unittest import mock

import rig_telemetry


class _LiveProcess:
    def is_alive(self):
        return True


class _SendSocket:
    def __init__(self, error=None):
        self.error = error
        self.sent = []
        self.closed = False

    def send(self, value):
        if self.error is not None:
            raise self.error
        self.sent.append(value)
        return len(value)

    def close(self):
        self.closed = True


class _FakeLocalSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.closed = False

    def settimeout(self, value):
        del value

    def recv(self, size):
        del size
        if self.messages:
            return self.messages.pop(0)
        raise OSError("test socket closed")

    def close(self):
        self.closed = True


class RigTelemetryTests(unittest.TestCase):
    def test_disabled_publisher_is_safe_noop(self):
        publisher = rig_telemetry.TelemetryPublisher(enabled=False)
        self.assertFalse(publisher.start())
        self.assertFalse(publisher.publish_state({"phase": "PRE"}))
        self.assertIsNone(publisher.parent_sock)
        publisher.close()

    def test_start_uses_spawn_socketpair_and_no_queue(self):
        process = mock.Mock()
        process.is_alive.return_value = True
        context = mock.Mock()
        context.Process.return_value = process
        publisher = rig_telemetry.TelemetryPublisher(context=context)
        try:
            self.assertTrue(publisher.start())
            self.assertIsNotNone(publisher.parent_sock)
            self.assertFalse(hasattr(publisher, "message_queue"))
            context.Queue.assert_not_called()
            context.Process.assert_called_once()
            process_call_args, process_call_kwargs = context.Process.call_args
            self.assertEqual(
                process_call_kwargs["target"],
                rig_telemetry._telemetry_worker,
            )
            process.start.assert_called_once_with()
        finally:
            publisher.close()

    def test_publish_is_immediate_local_datagram_handoff(self):
        parent_sock, child_sock = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_DGRAM
        )
        publisher = rig_telemetry.TelemetryPublisher(enabled=True)
        publisher.parent_sock = parent_sock
        publisher.process = _LiveProcess()
        try:
            started = time.monotonic()
            self.assertTrue(publisher.publish_state({"phase": "PRE", "trial": 1}))
            self.assertLess(time.monotonic() - started, 0.1)
            child_sock.settimeout(0.2)
            message = pickle.loads(child_sock.recv(4096))
            self.assertEqual(message["type"], "state")
            self.assertEqual(message["payload"], {"phase": "PRE", "trial": 1})
        finally:
            publisher.close()
            child_sock.close()

    def test_socket_pressure_drops_without_blocking(self):
        publisher = rig_telemetry.TelemetryPublisher(enabled=True)
        publisher.parent_sock = _SendSocket(BlockingIOError("full"))
        publisher.process = _LiveProcess()
        started = time.monotonic()
        self.assertFalse(publisher.publish_state({"phase": "PRE"}))
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertEqual(publisher.dropped_count, 1)

    def test_parent_oserror_drops_without_raising(self):
        publisher = rig_telemetry.TelemetryPublisher(enabled=True)
        publisher.parent_sock = _SendSocket(OSError("closed"))
        publisher.process = _LiveProcess()
        self.assertFalse(publisher.publish_state({"phase": "PRE"}))
        self.assertEqual(publisher.dropped_count, 1)

    def test_oversized_local_payload_is_dropped(self):
        publisher = rig_telemetry.TelemetryPublisher(enabled=True)
        sock = _SendSocket()
        publisher.parent_sock = sock
        publisher.process = _LiveProcess()
        self.assertFalse(
            publisher.publish_state({"payload": "x" * rig_telemetry.MAX_LOCAL_MESSAGE_SIZE})
        )
        self.assertEqual(publisher.dropped_count, 1)
        self.assertEqual(sock.sent, [])

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

    def test_worker_drops_malformed_packet_and_continues_to_shutdown(self):
        local_sock = _FakeLocalSocket([
            b"not pickle",
            pickle.dumps({"control": "shutdown"}, protocol=pickle.HIGHEST_PROTOCOL),
        ])
        udp_sock = mock.Mock()
        with mock.patch.object(rig_telemetry.socket, "socket", return_value=udp_sock):
            rig_telemetry._telemetry_worker(
                local_sock, "127.0.0.1", 5055, "session", "reward_conditioning"
            )
        udp_sock.sendto.assert_not_called()
        udp_sock.close.assert_called_once_with()
        self.assertTrue(local_sock.closed)

    def test_worker_send_failure_does_not_escape(self):
        local_sock = _FakeLocalSocket([
            pickle.dumps(
                {"type": "state", "payload": {"phase": "PRE"}},
                protocol=pickle.HIGHEST_PROTOCOL,
            ),
            pickle.dumps({"control": "shutdown"}, protocol=pickle.HIGHEST_PROTOCOL),
        ])
        udp_sock = mock.Mock()
        udp_sock.sendto.side_effect = OSError("unreachable")
        with mock.patch.object(rig_telemetry.socket, "socket", return_value=udp_sock):
            rig_telemetry._telemetry_worker(
                local_sock, "127.0.0.1", 5055, "session", "reward_conditioning"
            )
        udp_sock.close.assert_called_once_with()

    def test_close_terminates_stalled_worker_bounded(self):
        publisher = rig_telemetry.TelemetryPublisher(enabled=True, join_timeout_sec=0.01)
        process = mock.Mock()
        process.is_alive.side_effect = [True, True]
        publisher.parent_sock = _SendSocket(BlockingIOError("full"))
        publisher.process = process
        started = time.monotonic()
        publisher.close()
        self.assertLess(time.monotonic() - started, 0.2)
        process.terminate.assert_called_once_with()
        self.assertIsNone(publisher.parent_sock)
        self.assertIsNone(publisher.process)


if __name__ == "__main__":
    unittest.main(verbosity=2)
