# RuView ESPectre + ESP-CSI Staged Research Pipeline

This directory implements the long-term data plane for one fixed room, one
person, and three ESP32-S3 boards. ESPectre is a temporary motion benchmark;
it is not a runtime dependency and none of its GPLv3 source is copied here.

The stages are:

1. ESPectre 2.8.0 MVS/ML A/B on node 1 for `IDLE/MOTION`.
2. Restore node 1, then deploy this controlled ESP-CSI sender and two receivers.
3. Accept Presence, Motion, and Posture model heads independently.
4. Collect and accept fall trials only after Posture passes.

`IDLE` never means `ABSENT`. A missing link, status mismatch, unaccepted
profile, or model/topology mismatch remains `UNKNOWN`.

## Pinned Inputs

- ESP-CSI commit: `8633d67152db2808f141cc1595970aa9cf406045`
- ESP-IDF: `5.5.0`
- `esp_wifi_sensing`: `0.1.1~2`
- Wire protocol: `2`
- Target: ESP32-S3, HT20
- Candidate probe rates: 100 Hz, then 50 Hz fallback

The official `csi_send`, `csi_recv`, and `wifi_sensing_demo` examples remain
unmodified behavior baselines.

## Hardware Order

Do not connect all three boards to the Mac as data devices.

1. Connect only node 1 and verify MAC `28:84:85:92:81:3c`.
2. Run the ESPectre MVS/ML benchmark and restore node 1's own 16 MB image.
3. Flash node 1 with `posture_tx`, then move it to independent power.
4. Connect node 2, verify MAC `e0:72:a1:fd:19:0c`, and flash `posture_rx`.
5. Prove the first link before connecting node 3.
6. Connect node 3, verify MAC `28:84:85:45:f1:28`, and flash its receiver.

LED identity is red for node 1, green for node 2, and blue for node 3.
Flash offsets always come from each build's `flasher_args.json`.

## ESPectre Motion Benchmark

The pinned upstream release is `2.8.0` at commit
`29e457a0cf4251d681905f0df60832988f2f7559`.

- S3 MVS factory image SHA-256:
  `447263c78066287c070f4100603744e04cf446d9bf2710afa9abe58910534fe4`
- S3 ML factory image SHA-256:
  `5becc1ce9bcb23233f7bcbc9b162058048bf44a53e59dd2eebb0b40c8c01d847`

Install the Native API adapter outside the repository:

```bash
python3 -m venv ~/.local/share/ruview/venvs/espectre-benchmark
~/.local/share/ruview/venvs/espectre-benchmark/bin/pip install \
  -r tools/ruview_posture/requirements-espectre.txt
```

Record one labelled trial without Home Assistant:

```bash
python -m tools.ruview_posture.espectre_recorder \
  --host espectre.local \
  --expected-mac 28:84:85:92:81:3c \
  --algorithm mvs \
  --trial-id mvs-moving-01 \
  --scenario moving \
  --expected-motion motion \
  --duration 120 \
  --output ~/.local/share/ruview/data/espectre/mvs-moving-01.jsonl
```

Evaluate both modes:

```bash
python -m tools.ruview_posture.espectre_evaluate \
  ~/.local/share/ruview/data/espectre/*.jsonl \
  --output ~/.local/share/ruview/reports/espectre-ab.json
```

The gate requires 95 percent motion recall, no more than 5 percent idle false
motion, entry P95 at or below one second, exit P95 at or below two seconds,
ten entry and ten exit trials, and no unexpected disconnect.

## Private ESP-CSI Build

Wi-Fi credentials stay in ignored local ESP-IDF defaults. CI uses placeholders
and proves compilation only. Build the private sender and receivers from the
same commit. Set `CONFIG_RVP_PROBE_RATE_HZ=100` first; rebuild all three at
50 Hz only if either link misses the 80 Hz field gate.

```bash
cd examples/ruview-posture/posture_tx
idf.py set-target esp32s3
idf.py build

cd ../posture_rx
SDKCONFIG_DEFAULTS=sdkconfig.defaults idf.py set-target esp32s3
SDKCONFIG_DEFAULTS=sdkconfig.defaults idf.py build

rm -rf build sdkconfig
SDKCONFIG_DEFAULTS='sdkconfig.defaults;sdkconfig.defaults.node3' \
  idf.py set-target esp32s3
SDKCONFIG_DEFAULTS='sdkconfig.defaults;sdkconfig.defaults.node3' \
  idf.py build
```

The v2 receiver status packet reports source build ID, role, MACs, channel,
bandwidth, selected probe rate, gain-lock state, controlled-probe validity,
and persistent reboot count. The service rejects CSI until both receivers
match the private firmware binding.

Generate the binding from the three build artifact directories:

```bash
python -m tools.ruview_posture.firmware_binding \
  --fork-commit "$(git rev-parse HEAD)" \
  --probe-rate-hz 100 \
  --node-1 '<node-1-artifact-dir>' \
  --node-2 '<node-2-artifact-dir>' \
  --node-3 '<node-3-artifact-dir>' \
  --output ~/.local/share/ruview/config/firmware-binding.json
```

## Topology

Start from
[topology.example.json](topology.example.json), measure the real room and board
coordinates, and keep the completed file outside Git. Finalization validates
the one-transmitter/two-receiver layout and computes `topology_id` from
canonical content:

```bash
python -m tools.ruview_posture.topology \
  ~/.local/share/ruview/config/topology-draft.json \
  --output ~/.local/share/ruview/config/topology.json
```

Changing coordinates, MACs, channel, bandwidth, environment description, or
probe rate changes `topology_id` and invalidates the model.

## Collection Service

The service can collect data before a model exists:

```bash
python -m tools.ruview_posture.service \
  --topology ~/.local/share/ruview/config/topology.json \
  --firmware-binding ~/.local/share/ruview/config/firmware-binding.json \
  --recordings-directory ~/.local/share/ruview/data/esp-csi
```

Endpoints:

- `GET /health`
- `GET /api/v1/posture/latest`
- `GET /api/v1/posture/events`
- `POST /api/v1/experiments/session/start`
- `POST /api/v1/experiments/session/stop`
- `POST /api/v1/experiments/session/cancel`
- `GET /api/v1/experiments/session/status`

Session labels are independent fields:

```json
{
  "occupancy": "present",
  "motion": "idle",
  "posture": "sitting",
  "event": "none",
  "zone_id": "zone-2",
  "fan_state": "off",
  "curtain_state": "on",
  "trial_id": "sitting-zone-2-01",
  "dataset_role": "train",
  "countdown_seconds": 5,
  "duration_seconds": 120
}
```

The recorder writes raw versioned I/Q datagrams at the UDP entry point. The
status API reports the countdown, labels, remaining time, per-link valid frame
counts, output path, and failure reason.

## Features and Three Heads

Raw recordings retain complete CSI bytes. Deterministic two-second windows
extract:

- full 64-bin normalized amplitude and sanitized phase differences for
  Presence and Posture;
- independently implemented empty-room NBVI calibration that fixes 12
  non-consecutive subcarriers, a baseline MVS threshold, and its measured
  false-positive rate for each receiver link;
- Hampel-filtered moving variance for Motion, always using the fixed
  calibration stored in and hash-bound to the model manifest;
- RSSI, gain, sequence loss, frame rate, structure stability, and cross-link
  consistency for quality and model evidence.

Presence (`absent/present`), Motion (`still/moving`), and Posture
(`standing/sitting/lying`) are separate heads. Each starts with grouped
logistic regression. A TCN replaces a head only when grouped validation macro
F1 improves by at least three points and the worst class does not regress.

Train:

```bash
python -m tools.ruview_posture.train \
  ~/.local/share/ruview/data/esp-csi/train/*.rvp \
  --output ~/.local/share/ruview/models/posture-v1 \
  --topology ~/.local/share/ruview/config/topology.json \
  --firmware-binding ~/.local/share/ruview/config/firmware-binding.json
```

## Blind Acceptance and Activation

Blind recordings must have `dataset_role=blind` and come from a later time
period. Evaluate profiles independently and in order:

```bash
python -m tools.ruview_posture.evaluate \
  ~/.local/share/ruview/data/esp-csi/blind/*.rvp \
  --model ~/.local/share/ruview/models/posture-v1 \
  --topology ~/.local/share/ruview/config/topology.json \
  --firmware-binding ~/.local/share/ruview/config/firmware-binding.json \
  --profile presence \
  --output ~/.local/share/ruview/reports/presence.json
```

Repeat for `motion`, then `posture`. Run `fall` only after Posture passes.
Atomically bind only passed reports to the model:

```bash
python -m tools.ruview_posture.activate \
  --model ~/.local/share/ruview/models/posture-v1 \
  --topology ~/.local/share/ruview/config/topology.json \
  --firmware-binding ~/.local/share/ruview/config/firmware-binding.json \
  --acceptance ~/.local/share/ruview/reports/presence.json \
               ~/.local/share/ruview/reports/motion.json \
               ~/.local/share/ruview/reports/posture.json \
  --output ~/.local/share/ruview/models/posture-v1/activation.json
```

Start accepted inference with `--model` and `--activation`. Without an
activation record the same model runs shadow-only and the API remains invalid.
RuView integration stays blocked until locked blind tests and the two-hour
stability report pass. Skeletons, person count, vital signs, and medical or
safety claims remain unsupported.
