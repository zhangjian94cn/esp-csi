"""Timed field gates for controlled-link baseline, TX-off, and recovery."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np


def _fetch(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=2) as response:
        return json.load(response)


def _countdown(seconds: int, action: str) -> None:
    for remaining in range(seconds, 0, -1):
        print(
            json.dumps(
                {
                    "state": "countdown",
                    "action": action,
                    "remaining_seconds": remaining,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        time.sleep(1)
    print(
        json.dumps(
            {"state": "action_now", "action": action},
            ensure_ascii=False,
        ),
        flush=True,
    )


def collect_samples(
    url: str,
    *,
    duration_seconds: float,
    interval_seconds: float,
    fetch: Callable[[str], dict[str, Any]] = _fetch,
) -> tuple[list[tuple[float, dict[str, Any]]], int]:
    started = time.monotonic()
    samples: list[tuple[float, dict[str, Any]]] = []
    failures = 0
    while time.monotonic() - started < duration_seconds:
        elapsed = time.monotonic() - started
        try:
            samples.append((elapsed, fetch(url)))
        except (OSError, URLError, json.JSONDecodeError):
            failures += 1
        time.sleep(interval_seconds)
    return samples, failures


def baseline_report(
    samples: list[tuple[float, dict[str, Any]]],
    *,
    request_failures: int,
    probe_rate_hz: int,
) -> dict[str, Any]:
    per_node: dict[int, list[dict[str, Any]]] = {2: [], 3: []}
    for _, payload in samples:
        for link in payload.get("link_quality", []):
            node = int(link.get("rx_node_id", 0))
            if node in per_node:
                per_node[node].append(link)

    minimum_fps = probe_rate_hz * 0.8
    links: dict[str, Any] = {}
    gates: dict[str, bool] = {"request_stream": request_failures == 0}
    for node, rows in per_node.items():
        fps = [
            float(row["fps"])
            for row in rows
            if row.get("fps") is not None
        ]
        loss = [
            float(row["loss_rate"])
            for row in rows
            if row.get("loss_rate") is not None
        ]
        structure = [
            float(row["structure_stability"])
            for row in rows
            if row.get("structure_stability") is not None
        ]
        valid_rate = (
            sum(bool(row.get("valid")) for row in rows) / len(rows)
            if rows
            else 0.0
        )
        link_gates = {
            "sample_coverage": len(rows) >= max(1, int(len(samples) * 0.95)),
            "valid_rate": valid_rate >= 0.95,
            "fps": bool(fps) and float(np.percentile(fps, 5)) >= minimum_fps,
            "loss": bool(loss) and float(np.percentile(loss, 95)) <= 0.05,
            "structure": bool(structure)
            and float(np.percentile(structure, 5)) >= 0.95,
        }
        gates.update(
            {f"node_{node}_{name}": value for name, value in link_gates.items()}
        )
        links[str(node)] = {
            "samples": len(rows),
            "valid_rate": valid_rate,
            "fps_p5": float(np.percentile(fps, 5)) if fps else None,
            "loss_p95": float(np.percentile(loss, 95)) if loss else None,
            "structure_p5": (
                float(np.percentile(structure, 5)) if structure else None
            ),
            "gates": link_gates,
        }
    return {
        "phase": "baseline",
        "probe_rate_hz": probe_rate_hz,
        "request_failures": request_failures,
        "links": links,
        "gates": gates,
        "passed": all(gates.values()),
    }


def transition_report(
    samples: list[tuple[float, dict[str, Any]]],
    *,
    request_failures: int,
    phase: str,
) -> dict[str, Any]:
    if phase not in {"tx_off", "reconnect"}:
        raise ValueError("transition phase must be tx_off or reconnect")
    latency: float | None = None
    for elapsed, payload in samples:
        links = payload.get("link_quality", [])
        both_valid = (
            len(links) == 2
            and {int(link.get("rx_node_id", 0)) for link in links} == {2, 3}
            and all(bool(link.get("valid")) for link in links)
        )
        reached = (
            not payload.get("valid", False) and not both_valid
            if phase == "tx_off"
            else both_valid
        )
        if reached:
            latency = elapsed
            break
    limit = 2.0 if phase == "tx_off" else 10.0
    gates = {
        "request_stream": request_failures == 0,
        "transition_observed": latency is not None,
        "latency": latency is not None and latency <= limit,
    }
    return {
        "phase": phase,
        "latency_seconds": latency,
        "latency_limit_seconds": limit,
        "request_failures": request_failures,
        "gates": gates,
        "passed": all(gates.values()),
    }


def run(args: argparse.Namespace) -> int:
    action = {
        "baseline": "保持三块板供电并离开主要活动区域",
        "tx_off": "断开 1 号发送板电源",
        "reconnect": "重新接通 1 号发送板电源",
    }[args.phase]
    _countdown(args.countdown, action)
    duration = (
        args.duration
        if args.duration is not None
        else 60.0
        if args.phase == "baseline"
        else 5.0
        if args.phase == "tx_off"
        else 12.0
    )
    samples, failures = collect_samples(
        args.url,
        duration_seconds=duration,
        interval_seconds=args.interval,
    )
    if args.phase == "baseline":
        report = baseline_report(
            samples,
            request_failures=failures,
            probe_rate_hz=args.probe_rate_hz,
        )
    else:
        report = transition_report(
            samples,
            request_failures=failures,
            phase=args.phase,
        )
    result = {
        "schema": "rvp-link-smoke-v1",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "sample_count": len(samples),
        **report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["passed"] else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default="http://127.0.0.1:3100/api/v1/posture/latest"
    )
    parser.add_argument(
        "--phase",
        choices=["baseline", "tx_off", "reconnect"],
        required=True,
    )
    parser.add_argument("--probe-rate-hz", type=int, choices=[50, 100], required=True)
    parser.add_argument("--countdown", type=int, default=5)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.countdown <= 60:
        raise SystemExit("--countdown must be between 0 and 60")
    if args.duration is not None and args.duration <= 0:
        raise SystemExit("--duration must be positive")
    if not 0.1 <= args.interval <= 2.0:
        raise SystemExit("--interval must be between 0.1 and 2 seconds")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
