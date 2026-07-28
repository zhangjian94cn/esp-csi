# RuView Posture ESP-CSI PoC

This experiment keeps Espressif's examples intact and adds a controlled
single-transmitter, two-receiver data plane for a Mac-side posture model.
It is a research experiment, not a medical or safety alarm.

## Pinned toolchain

- ESP-CSI commit: `8633d67152db2808f141cc1595970aa9cf406045`
- ESP-IDF: `5.5.0`
- `esp_wifi_sensing`: `0.1.1~2`
- Target: ESP32-S3

The official `csi_send`, `csi_recv`, and `wifi_sensing_demo` examples remain
the unmodified behavior baseline. The RuView applications are:

- `posture_tx`: node 1, real factory MAC, HT20, 50 Hz ESP-NOW broadcast.
- `posture_rx`: nodes 2 and 3, source-MAC filtering, CSI queue, sink discovery,
  versioned UDP I/Q stream, and one-second status packets.

## Private Wi-Fi configuration

Do not commit credentials. Create a local ignored `sdkconfig` through
`idf.py menuconfig`, or provide the ESP-IDF example Wi-Fi settings through an
untracked defaults file. CI builds with non-secret placeholder settings.

All three nodes join the same 2.4 GHz access point. The sender reads the
current AP channel and the receivers follow that channel. The receiver
firmware discovers the Mac through a broadcast packet and then sends CSI by
unicast, so the Mac IP address is not compiled into firmware.

## Build

Build each application in ESP-IDF 5.5.0:

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

Use each build's `build/flasher_args.json`. Do not reuse offsets from RuView
firmware or from another ESP-IDF project.

## Safe device order

1. Connect only node 1, confirm MAC `28:84:85:92:81:3c`, validate its 16 MB
   backup, then flash and verify `posture_tx`.
2. Disconnect node 1's data cable and power it separately.
3. Connect only node 2, confirm MAC `e0:72:a1:fd:19:0c`, then flash and verify
   `posture_rx` with node ID 2.
4. Prove the first controlled link before touching node 3.
5. Connect only node 3, confirm MAC `28:84:85:45:f1:28`, then flash the node 3
   receiver build.

The identity LED is red for node 1, green for node 2, and blue for node 3.
If the board does not use GPIO 48 for its WS2812, set `CONFIG_RVP_LED_GPIO`
for that board before flashing.

## Mac tools

Create an environment outside the repository:

```bash
python3 -m venv ~/.local/share/ruview/venvs/esp-csi-posture
~/.local/share/ruview/venvs/esp-csi-posture/bin/pip install \
  -r tools/ruview_posture/requirements.txt
```

Record one explicitly labelled trial:

```bash
python -m tools.ruview_posture.collector \
  --output data/private/session-a-standing-01.rvp \
  --session-id session-a \
  --trial-id standing-01 \
  --zone-id zone-1 \
  --label standing \
  --dataset-role train \
  --topology-id '<topology-id>' \
  --duration 120
```

The collector writes to a `.partial` file and atomically renames it only after
the session closes. Raw household CSI recordings, model artifacts, camera
labels, and Wi-Fi settings must remain outside Git.

Train and run the standalone service:

```bash
python -m tools.ruview_posture.train data/private/*.rvp \
  --output data/private/model-v1 \
  --topology-id '<topology-id>'

python -m tools.ruview_posture.service \
  --model data/private/model-v1 \
  --topology-id '<topology-id>'
```

The service exposes:

- `GET http://127.0.0.1:3100/health`
- `GET http://127.0.0.1:3100/api/v1/posture/latest`
- `GET http://127.0.0.1:3100/api/v1/posture/events`

RuView integration is blocked until the locked blind test and two-hour
stability test pass.
