"""Discover receiver nodes and atomically record raw CSI on the Mac."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import socket
import threading
import time

from .protocol import (
    CSI_PORT,
    DISCOVERY_PORT,
    CsiPacket,
    ProtocolError,
    StatusPacket,
    decode_packet,
    encode_discovery,
)
from .recording import RecordingWriter


class DiscoveryBroadcaster(threading.Thread):
    def __init__(self, *, sink_port: int, stop: threading.Event):
        super().__init__(name="rvp-discovery", daemon=True)
        self.sink_port = sink_port
        self.stop = stop
        self.nonce = secrets.randbits(32)

    def run(self) -> None:
        packet = encode_discovery(sink_port=self.sink_port, nonce=self.nonce)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            while not self.stop.is_set():
                sock.sendto(packet, ("255.255.255.255", DISCOVERY_PORT))
                self.stop.wait(2.0)


def collect(args: argparse.Namespace) -> int:
    if args.delay:
        print(json.dumps({"state": "countdown", "seconds": args.delay}))
        time.sleep(args.delay)

    metadata = {
        "schema": "rvp-recording-v1",
        "session_id": args.session_id,
        "trial_id": args.trial_id,
        "zone_id": args.zone_id,
        "label": args.label,
        "dataset_role": args.dataset_role,
        "topology_id": args.topology_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "camera_labels": str(args.camera_labels) if args.camera_labels else None,
    }
    stop = threading.Event()
    broadcaster = DiscoveryBroadcaster(sink_port=args.port, stop=stop)
    counts: Counter[int] = Counter()
    invalid = 0
    statuses: dict[int, StatusPacket] = {}
    deadline = time.monotonic() + args.duration if args.duration else None

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", args.port))
        sock.settimeout(0.5)
        broadcaster.start()
        try:
            with RecordingWriter(args.output, metadata) as writer:
                while deadline is None or time.monotonic() < deadline:
                    try:
                        datagram, _ = sock.recvfrom(2048)
                    except socket.timeout:
                        continue
                    try:
                        packet = decode_packet(datagram)
                    except ProtocolError:
                        invalid += 1
                        continue
                    now_ns = time.time_ns()
                    writer.append(now_ns, datagram)
                    if isinstance(packet, CsiPacket):
                        counts[packet.node_id] += 1
                    else:
                        statuses[packet.node_id] = packet
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            broadcaster.join(timeout=3)

    result = {
        "state": "completed",
        "output": str(args.output),
        "valid_csi_frames": dict(sorted(counts.items())),
        "invalid_datagrams": invalid,
        "nodes": {
            node: {
                "build_id": status.build_id,
                "channel": status.channel,
                "tx_mac": status.tx_mac,
                "rx_mac": status.rx_mac,
                "frames_dropped": status.frames_dropped,
            }
            for node, status in sorted(statuses.items())
        },
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if len(counts) >= args.required_links else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--zone-id", default="unknown")
    parser.add_argument(
        "--label",
        required=True,
        choices=[
            "absent",
            "standing",
            "sitting",
            "lying",
            "moving",
            "fall",
            "slow_lying",
        ],
    )
    parser.add_argument(
        "--dataset-role", choices=["train", "blind"], default="train"
    )
    parser.add_argument("--topology-id", required=True)
    parser.add_argument("--camera-labels", type=Path)
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--port", type=int, default=CSI_PORT)
    parser.add_argument("--required-links", type=int, default=2)
    return parser


def main() -> int:
    return collect(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
