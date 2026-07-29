from __future__ import annotations

import json
from pathlib import Path

from tools.ruview_posture.firmware_binding import (
    artifact_identity,
    create_binding,
)


def artifact(path: Path, marker: bytes) -> None:
    (path / "bootloader").mkdir(parents=True)
    (path / "partition_table").mkdir()
    (path / "bootloader" / "bootloader.bin").write_bytes(marker + b"boot")
    (path / "partition_table" / "partition-table.bin").write_bytes(
        marker + b"part"
    )
    (path / "app.bin").write_bytes(marker + b"app")
    (path / "flasher_args.json").write_text(
        json.dumps(
            {
                "flash_files": {
                    "0x0": "bootloader/bootloader.bin",
                    "0xa000": "partition_table/partition-table.bin",
                    "0x20000": "app.bin",
                }
            }
        )
    )


def test_binding_hashes_every_referenced_flash_file(tmp_path: Path) -> None:
    directories = {}
    for node in ("1", "2", "3"):
        directory = tmp_path / node
        artifact(directory, node.encode())
        directories[node] = directory
    binding = create_binding(
        fork_commit="abcdef1234567890",
        probe_rate_hz=100,
        node_directories=directories,
    )
    assert binding.build_ids == {
        "1": "abcdef123456",
        "2": "abcdef123456",
        "3": "abcdef123456",
    }
    assert binding.artifact_sha256["1"] == artifact_identity(directories["1"])
    assert len(set(binding.artifact_sha256.values())) == 3
