"""Standalone research service for live ESP-CSI posture inference."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import socket
import threading
import time
from typing import Any

from .fall import FallDetector
from .features import STEP_NS, WINDOW_NS, extract_window
from .inference import PosturePredictor
from .protocol import (
    CSI_PORT,
    DISCOVERY_PORT,
    CsiPacket,
    ProtocolError,
    StatusPacket,
    decode_packet,
    encode_discovery,
)
from .recording import RecordedPacket


class StableValue:
    def __init__(self, confirmations: int = 3):
        self.confirmations = confirmations
        self.current = "unknown"
        self.candidate = "unknown"
        self.count = 0

    def update(self, value: str) -> str:
        if value == self.current:
            self.candidate = value
            self.count = 0
            return self.current
        if value != self.candidate:
            self.candidate = value
            self.count = 1
        else:
            self.count += 1
        if self.count >= self.confirmations:
            self.current = value
            self.count = 0
        return self.current


class LiveState:
    def __init__(
        self,
        *,
        predictor: PosturePredictor,
        event_log: Path,
        required_nodes: tuple[int, int],
    ):
        self.predictor = predictor
        self.event_log = event_log
        self.required_nodes = required_nodes
        self.frames: dict[int, deque[RecordedPacket]] = defaultdict(
            lambda: deque(maxlen=256)
        )
        self.statuses: dict[int, tuple[int, StatusPacket]] = {}
        self.latest: dict[str, Any] = self._unknown("starting")
        self.events: deque[dict[str, Any]] = deque(maxlen=100)
        self.lock = threading.Lock()
        self.stable_posture = StableValue(3)
        self.stable_state = StableValue(3)
        self.fall = FallDetector(
            motion_threshold=predictor.manifest.fall_motion_threshold
        )

    def _unknown(self, reason: str) -> dict[str, Any]:
        return {
            "state": "unknown",
            "posture": "unknown",
            "fall_event": "none",
            "valid": False,
            "confidence": 0.0,
            "source": "esp_csi_local_posture_model",
            "model_id": self.predictor.manifest.model_id,
            "topology_id": self.predictor.manifest.topology_id,
            "reason": reason,
            "link_quality": [],
            "research_only": True,
        }

    def ingest(self, host_timestamp_ns: int, datagram: bytes) -> None:
        packet = decode_packet(datagram)
        with self.lock:
            if isinstance(packet, CsiPacket):
                self.frames[packet.node_id].append(
                    RecordedPacket(host_timestamp_ns, datagram, packet)
                )
            else:
                self.statuses[packet.node_id] = (host_timestamp_ns, packet)

    def process(self) -> None:
        now_ns = time.time_ns()
        start_ns = now_ns - WINDOW_NS
        with self.lock:
            records = [
                record
                for node in self.required_nodes
                for record in self.frames.get(node, ())
                if record.host_timestamp_ns >= start_ns
            ]
        try:
            window = extract_window(
                records,
                start_ns=start_ns,
                end_ns=now_ns,
                required_nodes=self.required_nodes,
                min_frames_per_link=10,
            )
        except ValueError as error:
            with self.lock:
                self.latest = self._unknown(str(error))
            return

        link_quality = [
            {
                "node_id": node,
                "fps": frames / 2.0,
                "loss_rate": loss,
                "valid": frames / 2.0 >= 40.0 and loss <= 0.05,
            }
            for node, frames, loss in zip(
                window.node_ids, window.frames_per_node, window.loss_per_node
            )
        ]
        if not all(link["valid"] for link in link_quality):
            latest = self._unknown("link_quality")
            latest["link_quality"] = link_quality
            with self.lock:
                self.latest = latest
            return

        posture, confidence = self.predictor.predict(window)
        posture = self.stable_posture.update(posture)
        if posture == "absent":
            state = "absent"
            public_posture = "unknown"
        elif posture == "unknown":
            state = "unknown"
            public_posture = "unknown"
        else:
            state = (
                "present_moving"
                if window.motion_energy >= self.predictor.manifest.motion_threshold
                else "present_still"
            )
            public_posture = posture
        state = self.stable_state.update(state)
        valid = state != "unknown" and posture != "unknown"
        decision = self.fall.update(
            now_ms=now_ns // 1_000_000,
            posture=public_posture,
            motion_energy=window.motion_energy,
        )
        latest = {
            "state": state,
            "posture": public_posture,
            "fall_event": decision.event,
            "valid": valid,
            "confidence": confidence,
            "source": "esp_csi_local_posture_model",
            "model_id": self.predictor.manifest.model_id,
            "topology_id": self.predictor.manifest.topology_id,
            "motion_energy": window.motion_energy,
            "link_quality": link_quality,
            "updated_at_ns": now_ns,
            "research_only": True,
        }
        with self.lock:
            previous_event = self.latest.get("fall_event")
            self.latest = latest
            if decision.event == "suspected" and previous_event != "suspected":
                event = {
                    "event": "fall_suspected",
                    "timestamp_ns": now_ns,
                    "model_id": self.predictor.manifest.model_id,
                    "confidence": confidence,
                }
                self.events.append(event)
                self.event_log.parent.mkdir(parents=True, exist_ok=True)
                with self.event_log.open("a") as output:
                    output.write(json.dumps(event, sort_keys=True) + "\n")

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.latest))

    def event_snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.events)


class Handler(BaseHTTPRequestHandler):
    state: LiveState

    def do_GET(self) -> None:
        if self.path == "/health":
            body = {"status": "ok", "valid": self.state.snapshot()["valid"]}
        elif self.path == "/api/v1/posture/latest":
            body = self.state.snapshot()
        elif self.path == "/api/v1/posture/events":
            body = {"events": self.state.event_snapshot()}
        else:
            self.send_error(404)
            return
        encoded = json.dumps(body, sort_keys=True).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        return


def run(args: argparse.Namespace) -> int:
    predictor = PosturePredictor(args.model, topology_id=args.topology_id)
    state = LiveState(
        predictor=predictor,
        event_log=args.event_log,
        required_nodes=tuple(args.nodes),
    )
    Handler.state = state
    stop = threading.Event()

    def receive() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", args.csi_port))
            sock.settimeout(0.5)
            while not stop.is_set():
                try:
                    datagram, _ = sock.recvfrom(2048)
                    state.ingest(time.time_ns(), datagram)
                except socket.timeout:
                    continue
                except ProtocolError:
                    continue

    def discover() -> None:
        packet = encode_discovery(
            sink_port=args.csi_port, nonce=secrets.randbits(32)
        )
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            while not stop.is_set():
                sock.sendto(packet, ("255.255.255.255", DISCOVERY_PORT))
                stop.wait(2.0)

    def process() -> None:
        while not stop.is_set():
            state.process()
            stop.wait(STEP_NS / 1_000_000_000.0)

    threads = [
        threading.Thread(target=receive, daemon=True),
        threading.Thread(target=discover, daemon=True),
        threading.Thread(target=process, daemon=True),
    ]
    for thread in threads:
        thread.start()
    server = ThreadingHTTPServer((args.bind, args.http_port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--topology-id", required=True)
    parser.add_argument("--nodes", type=int, nargs=2, default=(2, 3))
    parser.add_argument("--csi-port", type=int, default=CSI_PORT)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=3100)
    parser.add_argument(
        "--event-log", type=Path, default=Path("data/fall-events.jsonl")
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
