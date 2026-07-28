"""Validate full ESP32 flash backups before any posture firmware is flashed."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import struct

FLASH_SIZE = 16 * 1024 * 1024
PARTITION_ENTRY = struct.Struct("<HBBII16sI")
PARTITION_MAGIC = 0x50AA
TABLE_CANDIDATES = (0x8000, 0x9000, 0xA000)


@dataclass(frozen=True)
class Partition:
    type: int
    subtype: int
    offset: int
    size: int
    label: str
    flags: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_partition_table(image: bytes, table_offset: int) -> list[Partition]:
    partitions: list[Partition] = []
    for offset in range(table_offset, table_offset + 0x1000, PARTITION_ENTRY.size):
        raw = image[offset : offset + PARTITION_ENTRY.size]
        if len(raw) != PARTITION_ENTRY.size:
            break
        magic, type_, subtype, part_offset, size, label_raw, flags = (
            PARTITION_ENTRY.unpack(raw)
        )
        if magic == 0xFFFF:
            break
        if magic != PARTITION_MAGIC:
            if magic == 0xEBEB:
                continue
            break
        label = label_raw.split(b"\0", 1)[0].decode("ascii", errors="replace")
        partitions.append(
            Partition(type_, subtype, part_offset, size, label, flags)
        )
    return partitions


def validate(path: Path, expected_hash: str | None = None) -> dict:
    image = path.read_bytes()
    failures: list[str] = []
    actual_hash = hashlib.sha256(image).hexdigest()
    if len(image) != FLASH_SIZE:
        failures.append(f"size:{len(image)}")
    if expected_hash is not None and actual_hash != expected_hash:
        failures.append("sha256")

    tables = {
        offset: parse_partition_table(image, offset)
        for offset in TABLE_CANDIDATES
    }
    tables = {offset: entries for offset, entries in tables.items() if entries}
    if not tables:
        failures.append("partition_table")
        selected_offset = None
        partitions: list[Partition] = []
    else:
        selected_offset, partitions = max(
            tables.items(), key=lambda item: len(item[1])
        )

    ranges: list[tuple[int, int, str]] = []
    for partition in partitions:
        end = partition.offset + partition.size
        if partition.offset < 0 or end > len(image) or partition.size == 0:
            failures.append(f"partition_bounds:{partition.label}")
        ranges.append((partition.offset, end, partition.label))
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if previous[1] > current[0]:
            failures.append(f"partition_overlap:{previous[2]}:{current[2]}")

    if not any(partition.type == 0 for partition in partitions):
        failures.append("application_partition")
    return {
        "path": str(path),
        "size": len(image),
        "sha256": actual_hash,
        "partition_table_offset": selected_offset,
        "partitions": [asdict(partition) for partition in partitions],
        "failures": failures,
        "valid": not failures,
    }


def load_hash_manifest(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, filename = line.split(maxsplit=1)
        hashes[str(Path(filename).expanduser())] = digest
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--hash-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    expected = (
        load_hash_manifest(args.hash_manifest) if args.hash_manifest else {}
    )
    results = [
        validate(
            image,
            expected.get(str(image.expanduser()))
            or expected.get(str(image.resolve())),
        )
        for image in args.images
    ]
    report = {"schema": "rvp-backup-validation-v1", "images": results}
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if all(result["valid"] for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
