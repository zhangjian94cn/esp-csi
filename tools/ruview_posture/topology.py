"""Canonical private topology validation and deterministic identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


TOPOLOGY_SCHEMA = "rvp-topology-v1"
REQUIRED_NODE_IDS = {1, 2, 3}
VALID_ROLES = {"tx", "rx"}


class TopologyError(ValueError):
    """Raised when a topology cannot safely bind firmware or a model."""


def _normalize_mac(value: str) -> str:
    parts = value.lower().replace("-", ":").split(":")
    if len(parts) != 6 or any(
        len(part) != 2
        or any(character not in "0123456789abcdef" for character in part)
        for part in parts
    ):
        raise TopologyError(f"invalid MAC address {value!r}")
    return ":".join(parts)


def _finite_number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TopologyError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise TopologyError(f"{field} must be finite")
    return result


def normalize_topology(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != TOPOLOGY_SCHEMA:
        raise TopologyError(f"schema must be {TOPOLOGY_SCHEMA}")
    room_source = payload.get("room")
    if not isinstance(room_source, Mapping):
        raise TopologyError("room must be an object")
    room = {
        "width_m": _finite_number(room_source.get("width_m"), "room.width_m"),
        "length_m": _finite_number(
            room_source.get("length_m"), "room.length_m"
        ),
        "height_m": _finite_number(
            room_source.get("height_m"), "room.height_m"
        ),
    }
    if any(value <= 0 for value in room.values()):
        raise TopologyError("room dimensions must be positive")

    try:
        channel = int(payload["channel"])
        bandwidth_mhz = int(payload["bandwidth_mhz"])
        probe_rate_hz = int(payload["probe_rate_hz"])
    except (KeyError, TypeError, ValueError) as error:
        raise TopologyError(
            "channel, bandwidth_mhz and probe_rate_hz are required integers"
        ) from error
    if not 1 <= channel <= 14:
        raise TopologyError("channel must be a 2.4 GHz channel")
    if bandwidth_mhz != 20:
        raise TopologyError("the first posture PoC requires HT20")
    if probe_rate_hz not in {50, 100}:
        raise TopologyError("probe_rate_hz must be 50 or 100")

    node_source = payload.get("nodes")
    if not isinstance(node_source, list):
        raise TopologyError("nodes must be a list")
    nodes: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_macs: set[str] = set()
    for source in node_source:
        if not isinstance(source, Mapping):
            raise TopologyError("each node must be an object")
        node_id = int(source.get("node_id", 0))
        role = str(source.get("role", ""))
        mac = _normalize_mac(str(source.get("mac", "")))
        position_source = source.get("position")
        if not isinstance(position_source, Mapping):
            raise TopologyError(f"node {node_id} position must be an object")
        position = {
            axis: _finite_number(
                position_source.get(axis), f"node {node_id}.position.{axis}"
            )
            for axis in ("x_m", "y_m", "z_m")
        }
        if node_id in seen_ids or mac in seen_macs:
            raise TopologyError("node IDs and MAC addresses must be unique")
        if role not in VALID_ROLES:
            raise TopologyError(f"node {node_id} has invalid role")
        if not 0 <= position["x_m"] <= room["width_m"]:
            raise TopologyError(f"node {node_id} x coordinate is outside room")
        if not 0 <= position["y_m"] <= room["length_m"]:
            raise TopologyError(f"node {node_id} y coordinate is outside room")
        if not 0 <= position["z_m"] <= room["height_m"]:
            raise TopologyError(f"node {node_id} z coordinate is outside room")
        seen_ids.add(node_id)
        seen_macs.add(mac)
        nodes.append(
            {
                "node_id": node_id,
                "role": role,
                "mac": mac,
                "position": position,
            }
        )
    if seen_ids != REQUIRED_NODE_IDS:
        raise TopologyError("topology requires exactly nodes 1, 2 and 3")
    roles = {node["node_id"]: node["role"] for node in nodes}
    if roles != {1: "tx", 2: "rx", 3: "rx"}:
        raise TopologyError("roles must be node 1 tx and nodes 2/3 rx")
    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1 :]:
            distance = math.sqrt(
                sum(
                    (
                        left["position"][axis]
                        - right["position"][axis]
                    )
                    ** 2
                    for axis in ("x_m", "y_m", "z_m")
                )
            )
            if not 2.0 <= distance <= 5.0:
                raise TopologyError(
                    f"nodes {left['node_id']} and {right['node_id']} "
                    f"must be 2-5 metres apart, got {distance:.2f}"
                )

    zones_source = payload.get("zones", [])
    if not isinstance(zones_source, list):
        raise TopologyError("zones must be a list")
    zones = sorted({str(zone).strip() for zone in zones_source if str(zone).strip()})
    if len(zones) < 6:
        raise TopologyError("at least six labelled zones are required")
    environment_source = payload.get("environment", {})
    if not isinstance(environment_source, Mapping):
        raise TopologyError("environment must be an object")
    environment = {
        "fan": str(environment_source.get("fan", "unknown")),
        "curtain": str(environment_source.get("curtain", "unknown")),
    }
    return {
        "schema": TOPOLOGY_SCHEMA,
        "room": room,
        "channel": channel,
        "bandwidth_mhz": bandwidth_mhz,
        "probe_rate_hz": probe_rate_hz,
        "nodes": sorted(nodes, key=lambda node: node["node_id"]),
        "zones": zones,
        "environment": environment,
    }


def topology_id(payload: Mapping[str, Any]) -> str:
    normalized = normalize_topology(payload)
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def finalize_topology(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_topology(payload)
    return {**normalized, "topology_id": topology_id(normalized)}


def load_topology(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text())
    expected = source.get("topology_id")
    finalized = finalize_topology(source)
    if expected is None:
        raise TopologyError("topology_id is missing; finalize the topology first")
    if expected != finalized["topology_id"]:
        raise TopologyError("topology_id does not match canonical content")
    return finalized


def node_map(topology: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(node["node_id"]): dict(node) for node in topology["nodes"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    finalized = finalize_topology(json.loads(args.source.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(finalized, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(finalized["topology_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
