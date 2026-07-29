"""Atomically bind a model to passed profile reports and stability evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .acceptance import PROFILES, file_sha256, load_acceptance
from .fall import FallModel
from .model import FirmwareBinding, ModelManifest
from .topology import load_topology


def activate(args: argparse.Namespace) -> int:
    topology = load_topology(args.topology)
    firmware = FirmwareBinding.load(args.firmware_binding)
    manifest_path = args.model / "manifest.json"
    manifest = ModelManifest.load(manifest_path)
    failures = manifest.validate_binding(
        topology_id=topology["topology_id"], firmware=firmware
    )
    if failures:
        raise ValueError("model binding mismatch: " + ", ".join(failures))
    accepted = load_acceptance(
        args.acceptance,
        model_id=manifest.model_id,
        topology_id=topology["topology_id"],
    )
    ordered_profiles = [profile for profile in PROFILES if profile in accepted]
    for index, profile in enumerate(ordered_profiles):
        if profile != PROFILES[index]:
            raise ValueError(
                "profiles must activate in order: presence, motion, posture, fall"
            )
    fall_model = FallModel.load(args.fall_model) if args.fall_model else None
    if "fall" in accepted:
        if fall_model is None:
            raise ValueError("fall activation requires --fall-model")
        fall_failures = fall_model.validate_binding(
            posture_model_id=manifest.model_id,
            topology_id=topology["topology_id"],
            motion_calibration_id=manifest.calibration_id,
        )
        if fall_failures:
            raise ValueError(
                "fall model binding mismatch: " + ", ".join(fall_failures)
            )
        if accepted["fall"].get("fall_model_id") != fall_model.fall_model_id:
            raise ValueError("fall report belongs to a different fall model")
    elif fall_model is not None:
        raise ValueError("--fall-model requires a fall acceptance report")

    stability = json.loads(args.stability.read_text())
    if (
        stability.get("schema") != "rvp-stability-v2"
        or not stability.get("passed")
    ):
        raise ValueError("stability report did not pass")
    if stability.get("model_ids") != [manifest.model_id]:
        raise ValueError("stability report belongs to a different model")
    if stability.get("topology_ids") != [topology["topology_id"]]:
        raise ValueError("stability report belongs to a different topology")
    if stability.get("sources") != ["esp_csi_validation_model"]:
        raise ValueError("stability report was not produced in validation mode")
    missing_capabilities = set(ordered_profiles) - set(
        stability.get("required_capabilities", [])
    )
    if missing_capabilities:
        raise ValueError(
            "stability report did not require activated capabilities: "
            + ", ".join(sorted(missing_capabilities))
        )

    payload = {
        "schema": "rvp-activation-v2",
        "state": "active",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": manifest.model_id,
        "topology_id": topology["topology_id"],
        "manifest_sha256": file_sha256(manifest_path),
        "firmware_binding_sha256": file_sha256(args.firmware_binding),
        "fall_model_id": (
            fall_model.fall_model_id if fall_model is not None else None
        ),
        "fall_model_sha256": (
            file_sha256(args.fall_model) if args.fall_model else None
        ),
        "acceptance": {
            profile: accepted[profile] for profile in ordered_profiles
        },
        "acceptance_sha256": {
            path.stem: file_sha256(path) for path in args.acceptance
        },
        "stability": stability,
        "stability_sha256": file_sha256(args.stability),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "model_id": manifest.model_id,
                "activated_profiles": ordered_profiles,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--firmware-binding", type=Path, required=True)
    parser.add_argument("--fall-model", type=Path)
    parser.add_argument("--acceptance", type=Path, nargs="+", required=True)
    parser.add_argument("--stability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return activate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
