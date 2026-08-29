"""Best-effort, read-only UDP telemetry for the rig monitor.

The experiment process only places small Python payloads onto a bounded queue.
The spawned worker owns timestamping, JSON encoding, and all network I/O.
"""

from __future__ import print_function

import datetime
import json
import multiprocessing
import queue
import socket
import time


SCHEMA_VERSION = 1
DEFAULT_HOST = "192.168.1.150"
DEFAULT_PORT = 5055
DEFAULT_QUEUE_SIZE = 64
DEFAULT_JOIN_TIMEOUT_SEC = 0.35


def _timestamp_utc():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _telemetry_worker(message_queue, host, port, session_id, protocol):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
    except Exception:
        return

    try:
        while True:
            try:
                message = message_queue.get(True, 0.10)
            except queue.Empty:
                continue
            if message is None:
                break
            try:
                packet = {
                    "schema_version": SCHEMA_VERSION,
                    "type": message["type"],
                    "protocol": protocol,
                    "session_id": session_id,
                    "timestamp_utc": _timestamp_utc(),
                }
                packet.update(message.get("payload") or {})
                encoded = json.dumps(packet, separators=(",", ":"),
                                      sort_keys=True).encode("utf-8")
                sock.sendto(encoded, (host, int(port)))
            except (Exception,):
                # Telemetry is never allowed to report an exception to the
                # experiment process. Drop malformed/undeliverable packets.
                continue
    finally:
        try:
            sock.close()
        except Exception:
            pass


class TelemetryPublisher(object):
    """Nonblocking parent-side publisher with bounded best-effort delivery."""

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, session_id="",
                 protocol="reward_conditioning", enabled=True,
                 queue_size=DEFAULT_QUEUE_SIZE,
                 join_timeout_sec=DEFAULT_JOIN_TIMEOUT_SEC,
                 context=None):
        self.host = str(host)
        self.port = int(port)
        self.session_id = str(session_id)
        self.protocol = str(protocol)
        self.enabled = bool(enabled)
        self.queue_size = int(queue_size)
        self.join_timeout_sec = float(join_timeout_sec)
        self.context = context or multiprocessing.get_context("spawn")
        self.message_queue = None
        self.process = None
        self.dropped_count = 0
        self.parent_error_count = 0
        self.start_error = None

    @property
    def available(self):
        return bool(self.enabled and self.process is not None and self.process.is_alive())

    def start(self):
        if not self.enabled:
            return False
        try:
            self.message_queue = self.context.Queue(maxsize=self.queue_size)
            self.process = self.context.Process(
                target=_telemetry_worker,
                args=(self.message_queue, self.host, self.port,
                      self.session_id, self.protocol),
            )
            self.process.daemon = True
            self.process.start()
            return True
        except Exception as exc:
            self.start_error = "%s: %s" % (type(exc).__name__, exc)
            self.parent_error_count += 1
            if self.process is not None:
                try:
                    if self.process.is_alive():
                        self.process.terminate()
                except Exception:
                    pass
            self.enabled = False
            self.process = None
            self.message_queue = None
            return False

    def _publish(self, message_type, payload):
        try:
            if not self.available:
                return False
            self.message_queue.put_nowait({
                "type": str(message_type),
                "payload": dict(payload or {}),
            })
            return True
        except queue.Full:
            self.dropped_count += 1
            return False
        except (OSError, ValueError, TypeError):
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
        message_queue = self.message_queue
        if process is None:
            return
        try:
            message_queue.put_nowait(None)
        except (queue.Full, OSError, ValueError, TypeError):
            pass
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
        try:
            message_queue.close()
        except Exception:
            pass
        self.process = None
