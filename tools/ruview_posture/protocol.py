"""Versioned wire protocol shared with the ESP32-S3 posture firmware."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Union

PROTOCOL_VERSION = 1
CSI_MAGIC = 0x31505652
STATUS_MAGIC = 0x53505652
DISCOVERY_MAGIC = 0x44505652
CSI_PORT = 5006
DISCOVERY_PORT = 5007
MAX_CSI_BYTES = 512

CSI_HEADER = struct.Struct("<IBBBBIQ6s6sBBbbbBHHhI")
STATUS = struct.Struct("<IBBBBI6s6sBBHIIII16sI")
DISCOVERY = struct.Struct("<IBBHII")

assert CSI_HEADER.size == 48
assert STATUS.size == 64
assert DISCOVERY.size == 16


class ProtocolError(ValueError):
    """Raised when a UDP datagram violates the protocol contract."""


def format_mac(raw: bytes) -> str:
    if len(raw) != 6:
        raise ProtocolError(f"MAC must contain 6 bytes, got {len(raw)}")
    return ":".join(f"{part:02x}" for part in raw)


@dataclass(frozen=True)
class CsiPacket:
    node_id: int
    role: int
    sequence: int
    timestamp_us: int
    tx_mac: str
    rx_mac: str
    channel: int
    bandwidth: int
    rssi: int
    noise_floor: int
    fft_gain: int
    agc_gain: int
    flags: int
    gain_compensation: float
    uptime_ms: int
    csi: bytes


@dataclass(frozen=True)
class StatusPacket:
    node_id: int
    role: int
    uptime_ms: int
    tx_mac: str
    rx_mac: str
    channel: int
    bandwidth: int
    flags: int
    frames_received: int
    frames_sent: int
    frames_dropped: int
    last_sequence: int
    build_id: str


Packet = Union[CsiPacket, StatusPacket]


def decode_packet(datagram: bytes) -> Packet:
    if len(datagram) < 8:
        raise ProtocolError("datagram is shorter than the common header")
    magic, version, packet_type, _, _ = struct.unpack_from("<IBBBB", datagram)
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")

    if magic == CSI_MAGIC and packet_type == 1:
        if len(datagram) < CSI_HEADER.size:
            raise ProtocolError("truncated CSI header")
        values = CSI_HEADER.unpack_from(datagram)
        csi_len = values[15]
        if csi_len > MAX_CSI_BYTES:
            raise ProtocolError(f"CSI payload exceeds {MAX_CSI_BYTES} bytes")
        if len(datagram) != CSI_HEADER.size + csi_len:
            raise ProtocolError(
                f"CSI length mismatch header={csi_len} datagram={len(datagram)}"
            )
        return CsiPacket(
            node_id=values[3],
            role=values[4],
            sequence=values[5],
            timestamp_us=values[6],
            tx_mac=format_mac(values[7]),
            rx_mac=format_mac(values[8]),
            channel=values[9],
            bandwidth=values[10],
            rssi=values[11],
            noise_floor=values[12],
            fft_gain=values[13],
            agc_gain=values[14],
            flags=values[16],
            gain_compensation=values[17] / 256.0,
            uptime_ms=values[18],
            csi=datagram[CSI_HEADER.size :],
        )

    if magic == STATUS_MAGIC and packet_type == 2:
        if len(datagram) != STATUS.size:
            raise ProtocolError(
                f"status packet must be {STATUS.size} bytes, got {len(datagram)}"
            )
        values = STATUS.unpack(datagram)
        return StatusPacket(
            node_id=values[3],
            role=values[4],
            uptime_ms=values[5],
            tx_mac=format_mac(values[6]),
            rx_mac=format_mac(values[7]),
            channel=values[8],
            bandwidth=values[9],
            flags=values[10],
            frames_received=values[11],
            frames_sent=values[12],
            frames_dropped=values[13],
            last_sequence=values[14],
            build_id=values[15].split(b"\0", 1)[0].decode("ascii", errors="replace"),
        )

    raise ProtocolError(
        f"unknown packet magic=0x{magic:08x} type={packet_type}"
    )


def encode_discovery(*, sink_port: int, nonce: int, ttl_ms: int = 10_000) -> bytes:
    if not 1 <= sink_port <= 65535:
        raise ValueError("sink_port must be between 1 and 65535")
    if not 2_000 <= ttl_ms <= 60_000:
        raise ValueError("ttl_ms must be between 2000 and 60000")
    return DISCOVERY.pack(
        DISCOVERY_MAGIC,
        PROTOCOL_VERSION,
        0,
        sink_port,
        nonce & 0xFFFFFFFF,
        ttl_ms,
    )
