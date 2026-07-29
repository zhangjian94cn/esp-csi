"""Record ESPectre Native API motion telemetry without Home Assistant."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time
from typing import Any

from .espectre_recording import EspectreRecordingWriter, EspectreSample


ESPHOME_PORT = 6053
EXPECTED_PROJECT = "francescopace.espectre"
ENTITY_NAMES = {
    "motion": "motion detected",
    "movement_score": "movement score",
    "threshold": "threshold",
}


@dataclass
class Telemetry:
    connected: bool = False
    motion: bool | None = None
    movement_score: float | None = None
    threshold: float | None = None
    event: str = "starting"


def normalize_mac(value: str) -> str:
    normalized = value.lower().replace(":", "").replace("-", "")
    if len(normalized) != 12 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("expected MAC must contain 12 hexadecimal digits")
    return normalized


def _entity_keys(entities: list[Any]) -> dict[str, int]:
    keys: dict[str, int] = {}
    for entity in entities:
        name = str(getattr(entity, "name", "")).strip().lower()
        object_id = str(getattr(entity, "object_id", "")).strip().lower()
        for field, expected in ENTITY_NAMES.items():
            if name == expected or object_id.endswith(expected.replace(" ", "_")):
                keys[field] = int(entity.key)
    missing = sorted(set(ENTITY_NAMES) - set(keys))
    if missing:
        raise RuntimeError(
            "ESPectre API is missing required entities: " + ", ".join(missing)
        )
    return keys


async def record(args: argparse.Namespace) -> int:
    try:
        from aioesphomeapi import APIClient
    except ImportError as error:
        raise RuntimeError(
            "install tools/ruview_posture/requirements-espectre.txt"
        ) from error

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop.set)
        except NotImplementedError:
            pass

    password = os.environ.get(args.password_env)
    noise_psk = os.environ.get(args.noise_psk_env)
    expected_mac = normalize_mac(args.expected_mac)
    metadata = {
        "algorithm": args.algorithm,
        "firmware_release": "2.8.0",
        "firmware_commit": "29e457a0cf4251d681905f0df60832988f2f7559",
        "expected_mac": expected_mac,
        "trial_id": args.trial_id,
        "scenario": args.scenario,
        "expected_motion": args.expected_motion == "motion",
        "transition": args.transition,
        "dataset_role": args.dataset_role,
        "warmup_seconds": args.warmup,
        "duration_seconds": args.duration,
    }
    telemetry = Telemetry()
    state_keys: dict[int, str] = {}
    disconnected_count = 0
    connected_at = 0.0
    deadline = time.monotonic() + args.duration

    async def on_stop(expected_disconnect: bool) -> None:
        nonlocal disconnected_count
        telemetry.connected = False
        telemetry.event = (
            "expected_disconnect" if expected_disconnect else "disconnected"
        )
        if not expected_disconnect:
            disconnected_count += 1

    client = APIClient(
        args.host,
        args.port,
        password=password,
        client_info="ruview-espectre-benchmark",
        noise_psk=noise_psk,
        expected_name=args.expected_name,
        expected_mac=expected_mac,
    )
    with EspectreRecordingWriter(args.output, metadata) as writer:
        try:
            await client.connect(on_stop=on_stop, login=True)
            info, entities, _ = await client.device_info_and_list_entities()
            if info.project_name != EXPECTED_PROJECT:
                raise RuntimeError(
                    f"unexpected ESPHome project {info.project_name!r}"
                )
            if normalize_mac(info.mac_address) != expected_mac:
                raise RuntimeError("connected device MAC does not match node 1")
            if info.project_version not in {"2.8.0", "main"}:
                raise RuntimeError(
                    f"unexpected ESPectre version {info.project_version!r}"
                )
            keys = _entity_keys(entities)
            state_keys = {key: field for field, key in keys.items()}
            telemetry.connected = True
            telemetry.event = "connected"
            connected_at = time.monotonic()

            def on_state(state: Any) -> None:
                field = state_keys.get(int(state.key))
                if field is None or bool(getattr(state, "missing_state", False)):
                    return
                if field == "motion":
                    telemetry.motion = bool(state.state)
                else:
                    setattr(telemetry, field, float(state.state))
                telemetry.event = f"{field}_update"

            client.subscribe_states(on_state)
            while not stop.is_set() and time.monotonic() < deadline:
                now = time.monotonic()
                writer.append(
                    EspectreSample(
                        monotonic_ns=time.monotonic_ns(),
                        wall_time=datetime.now(timezone.utc).isoformat(),
                        connected=telemetry.connected,
                        motion=telemetry.motion,
                        movement_score=telemetry.movement_score,
                        threshold=telemetry.threshold,
                        event=telemetry.event,
                    )
                )
                telemetry.event = "heartbeat"
                await asyncio.sleep(
                    max(0.0, args.sample_interval - (time.monotonic() - now))
                )
        finally:
            await client.disconnect()

    result = {
        "state": "completed",
        "output": str(args.output),
        "samples": writer.count,
        "connected_seconds": max(0.0, time.monotonic() - connected_at),
        "unexpected_disconnects": disconnected_count,
        "entities": state_keys,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if writer.count and disconnected_count == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=ESPHOME_PORT)
    parser.add_argument("--expected-mac", required=True)
    parser.add_argument("--expected-name", default="espectre")
    parser.add_argument("--algorithm", choices=["mvs", "ml"], required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument(
        "--scenario",
        choices=[
            "empty_fan_off",
            "environment_interference",
            "moving",
            "present_still",
            "enter",
            "exit",
        ],
        required=True,
    )
    parser.add_argument(
        "--expected-motion", choices=["idle", "motion"], required=True
    )
    parser.add_argument(
        "--transition", choices=["none", "enter", "exit"], default="none"
    )
    parser.add_argument(
        "--dataset-role", choices=["benchmark", "repeat"], default="benchmark"
    )
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--warmup", type=float, default=0.0)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--password-env", default="ESPECTRE_API_PASSWORD"
    )
    parser.add_argument("--noise-psk-env", default="ESPECTRE_NOISE_PSK")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    if not 0.05 <= args.sample_interval <= 1.0:
        raise SystemExit("--sample-interval must be between 0.05 and 1 second")
    return asyncio.run(record(args))


if __name__ == "__main__":
    raise SystemExit(main())
