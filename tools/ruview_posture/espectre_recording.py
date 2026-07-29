"""Atomic JSONL recordings for the temporary ESPectre motion benchmark."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping


SCHEMA = "rvp-espectre-motion-v1"


@dataclass(frozen=True)
class EspectreSample:
    monotonic_ns: int
    wall_time: str
    connected: bool
    motion: bool | None
    movement_score: float | None
    threshold: float | None
    event: str


class EspectreRecordingWriter(
    AbstractContextManager["EspectreRecordingWriter"]
):
    """Write metadata and samples, exposing only a completed atomic file."""

    def __init__(self, output: Path, metadata: Mapping[str, Any]):
        self.output = output
        self.partial = output.with_suffix(output.suffix + ".partial")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.partial.open("w", encoding="utf-8")
        header = {
            "record_type": "metadata",
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **dict(metadata),
        }
        self._write(header)
        self.count = 0

    def _write(self, payload: Mapping[str, Any]) -> None:
        self._file.write(json.dumps(dict(payload), sort_keys=True) + "\n")

    def append(self, sample: EspectreSample) -> None:
        self._write({"record_type": "sample", **asdict(sample)})
        self.count += 1

    def close(self) -> None:
        if self._file.closed:
            return
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        os.replace(self.partial, self.output)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            if not self._file.closed:
                self._file.close()
            self.partial.unlink(missing_ok=True)


def read_espectre_recording(
    path: Path,
) -> tuple[dict[str, Any], list[EspectreSample]]:
    metadata: dict[str, Any] | None = None
    samples: list[EspectreSample] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number} contains invalid JSON"
                ) from error
            record_type = payload.pop("record_type", None)
            if record_type == "metadata":
                if metadata is not None:
                    raise ValueError(f"{path} contains multiple metadata rows")
                if payload.get("schema") != SCHEMA:
                    raise ValueError(f"{path} uses an unsupported schema")
                metadata = payload
            elif record_type == "sample":
                samples.append(EspectreSample(**payload))
            else:
                raise ValueError(
                    f"{path}:{line_number} has unknown record_type"
                )
    if metadata is None:
        raise ValueError(f"{path} does not contain metadata")
    return metadata, samples


def iter_sample_dicts(path: Path) -> Iterator[dict[str, Any]]:
    """Yield plain dictionaries for report tooling."""

    _, samples = read_espectre_recording(path)
    for sample in samples:
        yield asdict(sample)
