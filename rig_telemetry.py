"""Best-effort, read-only UDP telemetry for the rig monitor.

The experiment process serializes small messages and performs one nonblocking
local datagram send.  A spawned worker owns timestamping, JSON encoding, and
all network I/O so telemetry cannot delay the experiment loop.
"""

from __future__ import print_function

import datetime
import json
import multiprocessing
import pickle
import socket


SCHEMA_VERSION = 1
DEFAULT_HOST = "192.168.1.150"
DEFAULT_PORT = 5055
DEFAULT_JOIN_TIMEOUT_SEC = 0.35
MAX_LOCAL_MESSAGE_SIZE = 16 * 1024


def _timestamp_utc():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _telemetry_worker(local_sock, host, port, session_id, protocol):
    """Forward parent datagrams to UDP without allowing errors to escape."""
    udp_sock = None
    try:
        local_sock.settimeout(0.10)
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_sock.setblocking(False)
    except Exception:
        try:
            local_sock.close()
        except Exception:
            pass
        if udp_sock is not None:
            try:
                udp_sock.close()
            except Exception:
                pass
        return

    try:
        while True:
            try:
                raw_message = local_sock.recv(MAX_LOCAL_MESSAGE_SIZE)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                message = pickle.loads(raw_message)
                if not isinstance(message, dict):
                    continue
                if message.get("control") == "shutdown":
                    break
                packet = {
                    "schema_version": SCHEMA_VERSION,
                    "type": message["type"],
                    "protocol": protocol,
                    "session_id": session_id,
                    "timestamp_utc": _timestamp_utc(),
                }
                packet.update(message.get("payload") or {})
                encoded = json.dumps(
                    packet, separators=(",", ":"), sort_keys=True
                ).encode("utf-8")
                udp_sock.sendto(encoded, (host, int(port)))
            except Exception:
                # Malformed local data, JSON failures, and UDP failures are
                # intentionally isolated to the telemetry worker.
                continue
    finally:
        try:
            local_sock.close()
        except Exception:
            pass
        if udp_sock is not None:
            try:
                udp_sock.close()
            except Exception:
                pass


class TelemetryPublisher(object):
    """Nonblocking parent-side publisher with bounded best-effort delivery."""

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, session_id="",
                 protocol="reward_conditioning", enabled=True,
                 queue_size=None,
                 join_timeout_sec=DEFAULT_JOIN_TIMEOUT_SEC,
                 context=None):
        # queue_size remains accepted for source compatibility with earlier
        # callers; AF_UNIX SOCK_DGRAM provides the bounded local transport.
        del queue_size
        self.host = str(host)
        self.port = int(port)
        self.session_id = str(session_id)
        self.protocol = str(protocol)
        self.enabled = bool(enabled)
        self.join_timeout_sec = float(join_timeout_sec)
        self.context = context or multiprocessing.get_context("spawn")
        self.parent_sock = None
        self.process = None
        self.dropped_count = 0
        self.parent_error_count = 0
        self.start_error = None

    @property
    def available(self):
        try:
            return bool(
                self.enabled
                and self.parent_sock is not None
                and self.process is not None
                and self.process.is_alive()
            )
        except Exception:
            return False

    def start(self):
        if not self.enabled:
            return False

        parent_sock = None
        child_sock = None
        process = None
        try:
            parent_sock, child_sock = socket.socketpair(
                socket.AF_UNIX, socket.SOCK_DGRAM
            )
            parent_sock.setblocking(False)
            self.parent_sock = parent_sock
            process = self.context.Process(
                target=_telemetry_worker,
                args=(child_sock, self.host, self.port,
                      self.session_id, self.protocol),
            )
            process.daemon = True
            self.process = process
            process.start()
            child_sock.close()
            child_sock = None
            return True
        except Exception as exc:
            self.start_error = "%s: %s" % (type(exc).__name__, exc)
            self.parent_error_count += 1
            if child_sock is not None:
                try:
                    child_sock.close()
                except Exception:
                    pass
            if process is not None:
                try:
                    if process.is_alive():
                        process.terminate()
                except Exception:
                    pass
            if parent_sock is not None:
                try:
                    parent_sock.close()
                except Exception:
                    pass
            self.enabled = False
            self.parent_sock = None
            self.process = None
            return False

    def _publish(self, message_type, payload):
        try:
            if not self.available:
                return False
            encoded = pickle.dumps(
                {"type": str(message_type), "payload": dict(payload or {})},
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            if len(encoded) > MAX_LOCAL_MESSAGE_SIZE:
                self.dropped_count += 1
                return False
            sent = self.parent_sock.send(encoded)
            if sent != len(encoded):
                self.dropped_count += 1
                return False
            return True
        except (BlockingIOError, OSError):
            self.dropped_count += 1
            return False
        except Exception:
            self.parent_error_count += 1
            return False

    def publish_session(self, payload):
        return self._publish("session", payload)

    def publish_state(self, payload):
        return self._publish("state", payload)

    def publish_trial(self, payload):
        return self._publish("trial_complete", payload)

    def publish_session_state(self, payload):
        return self._publish("session", payload)

    def close(self):
        process = self.process
        parent_sock = self.parent_sock
        self.process = None
        self.parent_sock = None

        if parent_sock is not None:
            try:
                shutdown = pickle.dumps(
                    {"control": "shutdown"}, protocol=pickle.HIGHEST_PROTOCOL
                )
                if len(shutdown) <= MAX_LOCAL_MESSAGE_SIZE:
                    parent_sock.send(shutdown)
            except (BlockingIOError, OSError, ValueError, TypeError):
                pass
            except Exception:
                self.parent_error_count += 1
            try:
                parent_sock.close()
            except Exception:
                pass

        if process is None:
            return
        try:
            process.join(self.join_timeout_sec)
        except Exception:
            pass
        try:
            process_alive = process.is_alive()
        except Exception:
            process_alive = False
        if process_alive:
            try:
                process.terminate()
                process.join(0.05)
            except Exception:
                pass
