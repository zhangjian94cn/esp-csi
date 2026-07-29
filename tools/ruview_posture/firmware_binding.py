"""Create a model binding from verified private firmware flash artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .model import FirmwareBinding
from .protocol import PROTOCOL_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_identity(directory: Path) -> str:
    manifest_path = directory / "flasher_args.json"
    manifest = json.loads(manifest_path.read_text())
    flash_files = manifest.get("flash_files")
    if not isinstance(flash_files, dict) or not flash_files:
        raise ValueError(f"{manifest_path} has no flash_files")
    entries: list[dict[str, Any]] = []
    for offset, relative in sorted(
        flash_files.items(), key=lambda item: int(item[0], 0)
    ):
        path = directory / str(relative)
        if not path.is_file():
            raise ValueError(f"missing flash file {path}")
        entries.append(
            {
                "offset": int(offset, 0),
                "path": str(relative),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def create_binding(
    *,
    fork_commit: str,
    probe_rate_hz: int,
    node_directories: dict[str, Path],
) -> FirmwareBinding:
    if set(node_directories) != {"1", "2", "3"}:
        raise ValueError("artifact directories are required for nodes 1, 2 and 3")
    build_id = fork_commit[:12]
    binding = FirmwareBinding(
        fork_commit=fork_commit,
        protocol_version=PROTOCOL_VERSION,
        probe_rate_hz=probe_rate_hz,
        build_ids={node: build_id for node in node_directories},
        artifact_sha256={
            node: artifact_identity(directory)
            for node, directory in node_directories.items()
        },
    )
    binding.validate()
    return binding


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fork-commit", required=True)
    parser.add_argument("--probe-rate-hz", type=int, choices=[50, 100], required=True)
    parser.add_argument("--node-1", type=Path, required=True)
    parser.add_argument("--node-2", type=Path, required=True)
    parser.add_argument("--node-3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binding = create_binding(
        fork_commit=args.fork_commit,
        probe_rate_hz=args.probe_rate_hz,
        node_directories={
            "1": args.node_1,
            "2": args.node_2,
            "3": args.node_3,
        },
    )
    payload = {"schema": "rvp-firmware-binding-v1", **binding.__dict__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
