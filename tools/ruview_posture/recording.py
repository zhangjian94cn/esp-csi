"""Atomic recording format for raw posture CSI datagrams."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import struct
from typing import BinaryIO, Iterator, Mapping, Any

from .protocol import Packet, decode_packet

FILE_MAGIC = b"RVPREC1\n"
METADATA_LENGTH = struct.Struct("<I")
RECORD_HEADER = struct.Struct("<QH")
MAX_DATAGRAM_BYTES = 2048


@dataclass(frozen=True)
class RecordedPacket:
    host_timestamp_ns: int
    datagram: bytes
    packet: Packet


class RecordingWriter(AbstractContextManager["RecordingWriter"]):
    def __init__(self, output: Path, metadata: Mapping[str, Any]):
        self.output = output
        self.partial = output.with_suffix(output.suffix + ".partial")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            dict(metadata), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self._file: BinaryIO = self.partial.open("wb")
        self._file.write(FILE_MAGIC)
        self._file.write(METADATA_LENGTH.pack(len(encoded)))
        self._file.write(encoded)
        self.count = 0

    def append(self, host_timestamp_ns: int, datagram: bytes) -> None:
        if len(datagram) > MAX_DATAGRAM_BYTES:
            raise ValueError(
                f"datagram exceeds recording limit: {len(datagram)} bytes"
            )
        decode_packet(datagram)
        self._file.write(RECORD_HEADER.pack(host_timestamp_ns, len(datagram)))
        self._file.write(datagram)
        self.count += 1

    def close(self) -> None:
        if self._file.closed:
            return
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        os.replace(self.partial, self.output)

    def abort(self) -> None:
        if not self._file.closed:
            self._file.close()
        self.partial.unlink(missing_ok=True)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def _open_recording(path: Path) -> tuple[BinaryIO, dict[str, Any]]:
    source = path.open("rb")
    if source.read(len(FILE_MAGIC)) != FILE_MAGIC:
        source.close()
        raise ValueError(f"{path} is not an RVP recording")
    raw_length = source.read(METADATA_LENGTH.size)
    if len(raw_length) != METADATA_LENGTH.size:
        source.close()
        raise ValueError(f"{path} has a truncated metadata length")
    (length,) = METADATA_LENGTH.unpack(raw_length)
    encoded = source.read(length)
    if len(encoded) != length:
        source.close()
        raise ValueError(f"{path} has truncated metadata")
    try:
        metadata = json.loads(encoded)
    except json.JSONDecodeError:
        source.close()
        raise
    return source, metadata


def read_metadata(path: Path) -> dict[str, Any]:
    source, metadata = _open_recording(path)
    source.close()
    return metadata


def iter_recording(path: Path) -> Iterator[RecordedPacket]:
    source, _ = _open_recording(path)
    try:
        while True:
            raw_header = source.read(RECORD_HEADER.size)
            if not raw_header:
                return
            if len(raw_header) != RECORD_HEADER.size:
                raise ValueError(f"{path} has a truncated record header")
            host_timestamp_ns, length = RECORD_HEADER.unpack(raw_header)
            if length > MAX_DATAGRAM_BYTES:
                raise ValueError(f"{path} contains an oversized record")
            datagram = source.read(length)
            if len(datagram) != length:
                raise ValueError(f"{path} has a truncated datagram")
            yield RecordedPacket(
                host_timestamp_ns=host_timestamp_ns,
                datagram=datagram,
                packet=decode_packet(datagram),
            )
    finally:
        source.close()
