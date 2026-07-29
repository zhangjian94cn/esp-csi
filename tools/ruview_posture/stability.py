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
    topology_ids: set[str] = set()
    sources: set[str] = set()
    capability_failures = 0
    link_failures = 0
    while time.monotonic() - started < args.duration:
        try:
            with urlopen(args.url, timeout=2) as response:
                payload = json.load(response)
            samples += 1
            invalid += int(not payload.get("valid", False))
            if payload.get("model_id"):
                model_ids.add(str(payload["model_id"]))
            if payload.get("topology_id"):
                topology_ids.add(str(payload["topology_id"]))
            if payload.get("source"):
                sources.add(str(payload["source"]))
            capabilities = payload.get("capabilities", {})
            capability_failures += int(
                any(
                    not capabilities.get(capability, False)
                    for capability in args.require_capability
                )
            )
            link_failures += int(
                not payload.get("link_quality")
                or not all(
                    link.get("valid", False)
                    for link in payload["link_quality"]
                )
            )
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
        "topology_ids": sorted(topology_ids),
        "sources": sorted(sources),
        "capability_failures": capability_failures,
        "required_capabilities": sorted(set(args.require_capability)),
        "link_failures": link_failures,
        "passed": (
            elapsed >= args.duration * 0.99
            and failures == 0
            and len(model_ids) == 1
            and len(topology_ids) == 1
            and sources == {"esp_csi_local_posture_model"}
            and capability_failures == 0
            and link_failures == 0
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
    parser.add_argument(
        "--require-capability",
        action="append",
        choices=["presence", "motion", "posture", "fall"],
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    return monitor(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
