"""Two-hour standalone service stability monitor."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.error import URLError
from urllib.request import urlopen


def monitor(args: argparse.Namespace) -> int:
    started = time.monotonic()
    samples = 0
    failures = 0
    invalid = 0
    model_ids: set[str] = set()
    while time.monotonic() - started < args.duration:
        try:
            with urlopen(args.url, timeout=2) as response:
                payload = json.load(response)
            samples += 1
            invalid += int(not payload.get("valid", False))
            if payload.get("model_id"):
                model_ids.add(str(payload["model_id"]))
        except (OSError, URLError, json.JSONDecodeError):
            failures += 1
        time.sleep(args.interval)

    elapsed = time.monotonic() - started
    result = {
        "schema": "rvp-stability-v1",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": elapsed,
        "samples": samples,
        "request_failures": failures,
        "invalid_samples": invalid,
        "invalid_rate": invalid / samples if samples else 1.0,
        "model_ids": sorted(model_ids),
        "passed": (
            elapsed >= args.duration * 0.99
            and failures == 0
            and len(model_ids) == 1
            and invalid / samples <= 0.10
            if samples
            else False
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default="http://127.0.0.1:3100/api/v1/posture/latest"
    )
    parser.add_argument("--duration", type=float, default=7200.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    return monitor(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
