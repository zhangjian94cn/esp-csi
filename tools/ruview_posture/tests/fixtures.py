from __future__ import annotations

from ..protocol import (
    CSI_HEADER,
    CSI_FLAG_PROBE_VALID,
    CSI_MAGIC,
    PROTOCOL_VERSION,
    STATUS,
    STATUS_FLAG_GAIN_LOCKED,
    STATUS_FLAG_SINK_VALID,
    STATUS_FLAG_TX_PROBE_VALID,
    STATUS_MAGIC,
)


def csi_datagram(
    *,
    node_id: int,
    sequence: int,
    csi: bytes,
    timestamp_us: int = 1,
    rssi: int = -42,
) -> bytes:
    header = CSI_HEADER.pack(
        CSI_MAGIC,
        PROTOCOL_VERSION,
        1,
        node_id,
        2,
        sequence,
        timestamp_us,
        bytes.fromhex("28848592813c"),
        (
            bytes.fromhex("e072a1fd190c")
            if node_id == 2
            else bytes.fromhex("28848545f128")
        ),
        6,
        20,
        rssi,
        -96,
        2,
        24,
        len(csi),
        CSI_FLAG_PROBE_VALID,
        256,
        timestamp_us // 1000,
    )
    return header + csi


def status_datagram(
    *,
    node_id: int = 2,
    build_id: str = "8633d671-rvp1",
    probe_rate_hz: int = 100,
) -> bytes:
    encoded_build = build_id.encode("ascii")[:15].ljust(16, b"\0")
    return STATUS.pack(
        STATUS_MAGIC,
        PROTOCOL_VERSION,
        2,
        node_id,
        2,
        5000,
        bytes.fromhex("28848592813c"),
        (
            bytes.fromhex("e072a1fd190c")
            if node_id == 2
            else bytes.fromhex("28848545f128")
        ),
        6,
        20,
        STATUS_FLAG_GAIN_LOCKED
        | STATUS_FLAG_SINK_VALID
        | STATUS_FLAG_TX_PROBE_VALID,
        250,
        245,
        5,
        249,
        encoded_build,
        probe_rate_hz,
        3,
    )
