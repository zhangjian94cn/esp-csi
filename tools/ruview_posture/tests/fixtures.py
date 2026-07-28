from __future__ import annotations

from pathlib import Path

from ..protocol import CSI_HEADER, CSI_MAGIC, STATUS, STATUS_MAGIC


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
        1,
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
        0,
        256,
        timestamp_us // 1000,
    )
    return header + csi


def status_datagram(*, node_id: int = 2) -> bytes:
    return STATUS.pack(
        STATUS_MAGIC,
        1,
        2,
        node_id,
        2,
        5000,
        bytes.fromhex("28848592813c"),
        bytes.fromhex("e072a1fd190c"),
        6,
        20,
        0,
        250,
        245,
        5,
        249,
        b"8633d671-rvp1\0\0",
        0,
    )
