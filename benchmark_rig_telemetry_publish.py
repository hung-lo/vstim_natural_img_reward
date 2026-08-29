#!/usr/bin/env python3
"""Measure parent-side telemetry handoff cost with a live child worker.

This is an operator diagnostic, not part of a normal experiment session.
"""

from __future__ import print_function

import argparse
import time

from rig_telemetry import TelemetryPublisher


def percentile(values, fraction):
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.1.150")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument(
        "--interval-ms", type=float, default=1.0,
        help="delay between calls so the live child can drain the local socket",
    )
    args = parser.parse_args()
    if args.count < 1 or args.interval_ms < 0:
        parser.error("--count must be at least 1 and --interval-ms nonnegative")

    publisher = TelemetryPublisher(
        host=args.host,
        port=args.port,
        session_id="telemetry-publish-benchmark",
    )
    if not publisher.start():
        raise SystemExit("telemetry worker failed to start: %s" % publisher.start_error)

    payload = {
        "phase": "STIMULUS",
        "trial": 32,
        "block": 1,
        "total_trials": 50,
        "stimulus": "representative_stimulus_filename.png",
        "reward_scheduled": True,
        "reward_delivered": False,
        "reward_contacted": None,
        "lick_times_sec": [0.143, 0.381, 0.724],
        "task_water_delivered_ul_session": 0.04,
    }
    durations_us = []
    try:
        # Let the spawned worker reach its receive loop before measuring the
        # steady-state local handoff rather than startup scheduling.
        time.sleep(0.10)
        for _ in range(args.count):
            started = time.perf_counter()
            publisher.publish_state(payload)
            durations_us.append((time.perf_counter() - started) * 1000000.0)
            if args.interval_ms:
                time.sleep(args.interval_ms / 1000.0)
    finally:
        publisher.close()

    print("publish calls: %d" % len(durations_us))
    print("dropped calls: %d" % publisher.dropped_count)
    print("parent errors: %d" % publisher.parent_error_count)
    print("median_us: %.3f" % percentile(durations_us, 0.50))
    print("p95_us: %.3f" % percentile(durations_us, 0.95))
    print("max_us: %.3f" % max(durations_us))


if __name__ == "__main__":
    main()
