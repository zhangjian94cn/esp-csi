"""Standalone ESP-CSI collection and staged posture inference service."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import socket
import threading
import time
from typing import Any

from .acceptance import capabilities, load_activation
from .experiment import ExperimentSessionManager, SessionConflict
from .fall import FallDetector
from .features import STEP_NS, WINDOW_NS, extract_window
from .inference import PosturePredictor
from .model import FirmwareBinding
from .protocol import (
    CSI_PORT,
    CSI_FLAG_PROBE_VALID,
    DISCOVERY_PORT,
    STATUS_FLAG_SINK_VALID,
    STATUS_FLAG_TX_PROBE_VALID,
    CsiPacket,
    ProtocolError,
    StatusPacket,
    decode_packet,
    encode_discovery,
)
from .recording import RecordedPacket
from .topology import load_topology, node_map


class StableValue:
    def __init__(self, confirmations: dict[str, int] | int = 3):
        self.confirmations = confirmations
        self.current = "unknown"
        self.candidate = "unknown"
        self.count = 0

    def update(self, value: str) -> str:
        if value == "unknown":
            self.current = "unknown"
            self.candidate = "unknown"
            self.count = 0
            return self.current
        if value == self.current:
            self.candidate = value
            self.count = 0
            return self.current
        if value != self.candidate:
            self.candidate = value
            self.count = 1
        else:
            self.count += 1
        required = (
            self.confirmations.get(value, 3)
            if isinstance(self.confirmations, dict)
            else self.confirmations
        )
        if self.count >= required:
            self.current = value
            self.count = 0
        return self.current


class LiveState:
    def __init__(
        self,
        *,
        predictor: PosturePredictor | None,
        topology: dict[str, Any],
        firmware: FirmwareBinding,
        acceptance: dict[str, dict[str, Any]],
        event_log: Path,
        recordings_directory: Path,
    ):
        self.predictor = predictor
        self.topology = topology
        self.firmware = firmware
        self.acceptance = acceptance
        self.capabilities = capabilities(acceptance)
        self.event_log = event_log
        nodes = node_map(topology)
        self.required_nodes = tuple(
            sorted(node for node, value in nodes.items() if value["role"] == "rx")
        )
        self.nodes = nodes
        self.frames: dict[int, deque[RecordedPacket]] = defaultdict(
            lambda: deque(maxlen=512)
        )
        self.statuses: dict[int, tuple[int, StatusPacket]] = {}
        self.reboot_counts: dict[int, int] = {}
        self.reboot_quarantine_until_ns: dict[int, int] = {}
        self.events: deque[dict[str, Any]] = deque(maxlen=100)
        self.lock = threading.Lock()
        self.stable_presence = StableValue({"present": 3, "absent": 4})
        self.stable_motion = StableValue(3)
        self.stable_posture = StableValue(3)
        self.fall = (
            FallDetector(
                motion_threshold=predictor.manifest.fall_motion_threshold
            )
            if predictor is not None
            else None
        )
        self.experiment = ExperimentSessionManager(
            output_directory=recordings_directory,
            topology_id=topology["topology_id"],
            required_nodes=self.required_nodes,
        )
        self.latest: dict[str, Any] = self._unknown("starting")

    @property
    def model_id(self) -> str | None:
        return self.predictor.manifest.model_id if self.predictor else None

    def _model_payload(self) -> dict[str, Any]:
        return {
            "loaded": self.predictor is not None,
            "model_id": self.model_id,
            "topology_id": self.topology["topology_id"],
            "accepted_profiles": sorted(self.acceptance),
            "probe_rate_hz": self.firmware.probe_rate_hz,
        }

    def _unknown(
        self, reason: str, link_quality: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        return {
            "state": "unknown",
            "posture": "unknown",
            "fall_event": "none",
            "valid": False,
            "confidence": 0.0,
            "source": (
                "esp_csi_shadow_model"
                if self.predictor
                else "esp_csi_raw_data_plane"
            ),
            "model_id": self.model_id,
            "topology_id": self.topology["topology_id"],
            "reason": reason,
            "capabilities": self.capabilities,
            "model": self._model_payload(),
            "link_quality": link_quality or [],
            "research_only": True,
        }

    def _packet_matches_topology(self, packet: CsiPacket) -> bool:
        expected = self.nodes.get(packet.node_id)
        tx = self.nodes[1]
        return bool(
            expected
            and expected["role"] == "rx"
            and packet.role == 2
            and packet.tx_mac == tx["mac"]
            and packet.rx_mac == expected["mac"]
            and packet.channel == self.topology["channel"]
            and packet.bandwidth == self.topology["bandwidth_mhz"]
            and bool(packet.flags & CSI_FLAG_PROBE_VALID)
        )

    def ingest(self, host_timestamp_ns: int, datagram: bytes) -> None:
        packet = decode_packet(datagram)
        monotonic_ns = time.monotonic_ns()
        self.experiment.ingest(
            host_timestamp_ns, monotonic_ns, datagram, packet
        )
        with self.lock:
            if isinstance(packet, CsiPacket):
                if self._packet_matches_topology(packet):
                    self.frames[packet.node_id].append(
                        RecordedPacket(host_timestamp_ns, datagram, packet)
                    )
                return
            if packet.node_id not in self.required_nodes:
                return
            previous = self.reboot_counts.get(packet.node_id)
            if previous is not None and previous != packet.reboot_count:
                self.reboot_quarantine_until_ns[packet.node_id] = (
                    host_timestamp_ns + 10_000_000_000
                )
            self.reboot_counts[packet.node_id] = packet.reboot_count
            self.statuses[packet.node_id] = (host_timestamp_ns, packet)

    def _status_quality_locked(
        self, now_ns: int
    ) -> tuple[list[str], list[dict[str, Any]]]:
        failures: list[str] = []
        quality: list[dict[str, Any]] = []
        tx = self.nodes[1]
        for node in self.required_nodes:
            expected = self.nodes[node]
            status_record = self.statuses.get(node)
            if status_record is None:
                failures.append(f"node_{node}_status_missing")
                quality.append(
                    {
                        "tx_node_id": 1,
                        "rx_node_id": node,
                        "valid": False,
                        "reason": "status_missing",
                    }
                )
                continue
            timestamp_ns, status = status_record
            stale_seconds = max(0.0, (now_ns - timestamp_ns) / 1_000_000_000)
            reasons: list[str] = []
            checks = {
                "stale": stale_seconds <= 2.0,
                "role": status.role == 2,
                "tx_mac": status.tx_mac == tx["mac"],
                "rx_mac": status.rx_mac == expected["mac"],
                "channel": status.channel == self.topology["channel"],
                "bandwidth": status.bandwidth
                == self.topology["bandwidth_mhz"],
                "probe_rate": status.probe_rate_hz
                == self.firmware.probe_rate_hz,
                "build_id": status.build_id
                == self.firmware.build_ids[str(node)],
                "gain_locked": status.gain_locked,
                "sink_valid": bool(status.flags & STATUS_FLAG_SINK_VALID),
                "tx_probe_valid": bool(
                    status.flags & STATUS_FLAG_TX_PROBE_VALID
                ),
                "reboot_quarantine": now_ns
                >= self.reboot_quarantine_until_ns.get(node, 0),
            }
            reasons.extend(name for name, passed in checks.items() if not passed)
            if reasons:
                failures.extend(f"node_{node}_{reason}" for reason in reasons)
            quality.append(
                {
                    "tx_node_id": 1,
                    "rx_node_id": node,
                    "status_age_seconds": stale_seconds,
                    "build_id": status.build_id,
                    "probe_rate_hz": status.probe_rate_hz,
                    "gain_locked": status.gain_locked,
                    "reboot_count": status.reboot_count,
                    "valid": not reasons,
                    "reasons": reasons,
                }
            )
        return failures, quality

    def topology_failures(self) -> list[str]:
        with self.lock:
            failures, _ = self._status_quality_locked(time.time_ns())
            return failures

    def process(self) -> None:
        now_ns = time.time_ns()
        start_ns = now_ns - WINDOW_NS
        with self.lock:
            status_failures, link_quality = self._status_quality_locked(now_ns)
            records = [
                record
                for node in self.required_nodes
                for record in self.frames.get(node, ())
                if record.host_timestamp_ns >= start_ns
            ]
        if status_failures:
            with self.lock:
                self.latest = self._unknown(
                    "status:" + ",".join(status_failures), link_quality
                )
            return

        try:
            window = extract_window(
                records,
                start_ns=start_ns,
                end_ns=now_ns,
                required_nodes=self.required_nodes,
                min_frames_per_link=10,
                motion_subcarriers=(
                    self.predictor.motion_calibration.subcarriers()
                    if self.predictor
                    else None
                ),
            )
        except ValueError as error:
            with self.lock:
                self.latest = self._unknown(str(error), link_quality)
            return

        minimum_fps = self.firmware.probe_rate_hz * 0.8
        for index, (node, frames, loss, structure) in enumerate(
            zip(
                window.node_ids,
                window.frames_per_node,
                window.loss_per_node,
                window.structure_stability_per_node,
            )
        ):
            fps = frames / 2.0
            valid = fps >= minimum_fps and loss <= 0.05 and structure >= 0.95
            link_quality[index].update(
                {
                    "rx_node_id": node,
                    "fps": fps,
                    "loss_rate": loss,
                    "structure_stability": structure,
                    "valid": valid,
                }
            )
        if not all(link["valid"] for link in link_quality):
            with self.lock:
                self.latest = self._unknown("link_quality", link_quality)
            return
        if self.predictor is None:
            with self.lock:
                self.latest = self._unknown("model_not_loaded", link_quality)
            return
        if not self.capabilities["presence"]:
            with self.lock:
                self.latest = self._unknown(
                    "presence_not_accepted", link_quality
                )
            return

        prediction = self.predictor.predict(window)
        presence = self.stable_presence.update(prediction.presence.label)
        if presence == "unknown":
            with self.lock:
                self.latest = self._unknown("presence_unknown", link_quality)
            return

        if presence == "absent":
            state = "absent"
            posture = "unknown"
            motion = "unknown"
        else:
            motion = (
                self.stable_motion.update(prediction.motion.label)
                if self.capabilities["motion"]
                else "unknown"
            )
            posture = (
                self.stable_posture.update(prediction.posture.label)
                if self.capabilities["posture"]
                else "unknown"
            )
            state = (
                "present_moving"
                if motion == "moving"
                else "present_still"
                if motion == "still"
                else "present_unknown"
            )

        fall_event = "none"
        if (
            self.capabilities["fall"]
            and self.fall is not None
            and presence == "present"
        ):
            fall_event = self.fall.update(
                now_ms=now_ns // 1_000_000,
                posture=posture,
                motion_energy=window.motion_energy,
            ).event
        latest = {
            "state": state,
            "posture": posture,
            "fall_event": fall_event,
            "valid": True,
            "confidence": prediction.confidence,
            "source": "esp_csi_local_posture_model",
            "model_id": self.model_id,
            "topology_id": self.topology["topology_id"],
            "motion_energy": window.motion_energy,
            "capabilities": self.capabilities,
            "model": self._model_payload(),
            "link_quality": link_quality,
            "updated_at_ns": now_ns,
            "research_only": True,
        }
        with self.lock:
            previous_event = self.latest.get("fall_event")
            self.latest = latest
            if fall_event == "suspected" and previous_event != "suspected":
                event = {
                    "event": "fall_suspected",
                    "timestamp_ns": now_ns,
                    "model_id": self.model_id,
                    "confidence": prediction.confidence,
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

    def _json(self, status: int, body: Any) -> None:
        encoded = json.dumps(body, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/health":
            snapshot = self.state.snapshot()
            body = {
                "status": "ok",
                "valid": snapshot["valid"],
                "reason": snapshot.get("reason"),
                "capabilities": snapshot["capabilities"],
            }
        elif self.path == "/api/v1/posture/latest":
            body = self.state.snapshot()
        elif self.path == "/api/v1/posture/events":
            body = {"events": self.state.event_snapshot()}
        elif self.path == "/api/v1/experiments/session/status":
            body = self.state.experiment.snapshot()
        else:
            self.send_error(404)
            return
        self._json(200, body)

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = (
                json.loads(self.rfile.read(content_length))
                if content_length
                else {}
            )
            if self.path == "/api/v1/experiments/session/start":
                failures = self.state.topology_failures()
                if failures:
                    raise SessionConflict(
                        "topology is not ready: " + ",".join(failures)
                    )
                body, created = self.state.experiment.start(payload)
                self._json(201 if created else 200, body)
            elif self.path == "/api/v1/experiments/session/stop":
                self._json(200, self.state.experiment.stop())
            elif self.path == "/api/v1/experiments/session/cancel":
                self._json(200, self.state.experiment.cancel())
            else:
                self.send_error(404)
        except (KeyError, TypeError, ValueError) as error:
            self._json(400, {"error": str(error)})
        except SessionConflict as error:
            self._json(409, {"error": str(error)})

    def log_message(self, format: str, *args: Any) -> None:
        return


def run(args: argparse.Namespace) -> int:
    topology = load_topology(args.topology)
    firmware = FirmwareBinding.load(args.firmware_binding)
    if topology["probe_rate_hz"] != firmware.probe_rate_hz:
        raise ValueError("topology and firmware probe rates do not match")
    predictor = (
        PosturePredictor(
            args.model,
            topology_id=topology["topology_id"],
            firmware=firmware,
        )
        if args.model
        else None
    )
    acceptance = (
        load_activation(
            args.activation,
            model_id=predictor.manifest.model_id,
            topology_id=topology["topology_id"],
        )
        if predictor and args.activation
        else {}
    )
    state = LiveState(
        predictor=predictor,
        topology=topology,
        firmware=firmware,
        acceptance=acceptance,
        event_log=args.event_log,
        recordings_directory=args.recordings_directory,
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
            state.experiment.snapshot()
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
    parser.add_argument("--model", type=Path)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--firmware-binding", type=Path, required=True)
    parser.add_argument("--activation", type=Path)
    parser.add_argument("--csi-port", type=int, default=CSI_PORT)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=3100)
    parser.add_argument(
        "--event-log", type=Path, default=Path("data/fall-events.jsonl")
    )
    parser.add_argument(
        "--recordings-directory",
        type=Path,
        default=Path("data/private/recordings"),
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
