"""Synthetic CPU benchmark for local iBeacon callback/coalescing overhead.

Run from the repository root:
    python tests/benchmark_local_ibeacon.py --packets 200000 --repeats 7

This excludes HA dispatch, radio scanning and normal distance calculations.
It measures process CPU time on synthetic already-created packet objects.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "bermuda_local_ibeacon", Path(__file__).resolve().parents[1] / "custom_components/bermuda/local_ibeacon.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
LocalIBeaconCapture = module.LocalIBeaconCapture

DATA = bytes.fromhex("021500112233445566778899aabbccddeeff00649c44c5")
IDENTITY = "00112233445566778899AABBCCDDEEFF_100_40004"
SOURCE = "11:22:33:44:55:66"
ADDRESS = "AA:BB:CC:DD:EE:01"


def measure(mode, packets):
    clock = [100.0]
    capture = LocalIBeaconCapture(
        lambda callback, address: lambda: None,
        lambda: [IDENTITY],
        lambda source: source == SOURCE,
        lambda: clock[0],
    )
    info = SimpleNamespace(
        address=ADDRESS,
        source=SOURCE,
        time=clock[0],
        rssi=-64,
        manufacturer_data={76: DATA},
        raw=bytes.fromhex("1aff4c00") + DATA,
    )
    capture.observe(info)
    capture.drain()
    started = time.process_time_ns()
    for index in range(packets):
        clock[0] += 1 / 3
        info.time = clock[0]
        if mode != "baseline":
            capture.capture(info)
        if mode == "capture_and_coalesce" and index % 3 == 2:
            capture.drain()
    elapsed = time.process_time_ns() - started
    pending = len(capture._pending)
    capture.close()
    assert pending <= 1
    return elapsed / packets / 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, default=200000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.packets < 1000 or args.repeats < 3:
        parser.error("Use at least 1000 packets and 3 repeats.")
    modes = ["baseline", "capture_only", "capture_and_coalesce"]
    for mode in modes:
        measure(mode, 10000)
    runs = {mode: [] for mode in modes}
    for repeat in range(args.repeats):
        # Rotate ordering to reduce systematic warmup/frequency bias.
        order = modes[repeat % len(modes) :] + modes[: repeat % len(modes)]
        for mode in order:
            runs[mode].append(measure(mode, args.packets))
    summary = {
        mode: {
            "median_cpu_us_per_packet": statistics.median(values),
            "min_cpu_us_per_packet": min(values),
            "max_cpu_us_per_packet": max(values),
        }
        for mode, values in runs.items()
    }
    baseline = summary["baseline"]["median_cpu_us_per_packet"]
    for mode in modes[1:]:
        net = summary[mode]["median_cpu_us_per_packet"] - baseline
        summary[mode]["net_cpu_us_per_packet"] = net
        summary[mode]["estimated_percent_one_core"] = {str(rate): net * rate / 10000 for rate in [3, 30, 300, 3000]}
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "machine": platform.machine(),
                "logical_cpus": os.cpu_count(),
                "packets_per_repeat": args.packets,
                "repeats": args.repeats,
                "clock": "synthetic monotonic time, advancing 1/3 second per packet",
                "coalescing": "one drain per three matching packets",
                "scope": "synthetic callback and queue/drain cost; excludes HA dispatch, radio and distance processing",
                "results": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
