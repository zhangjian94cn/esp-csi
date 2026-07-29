/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RVP_PROTOCOL_VERSION 2U
#define RVP_CSI_MAGIC 0x31505652UL
#define RVP_STATUS_MAGIC 0x53505652UL
#define RVP_DISCOVERY_MAGIC 0x44505652UL
#define RVP_PROBE_MAGIC 0x42505652UL
#define RVP_CSI_PORT 5006U
#define RVP_DISCOVERY_PORT 5007U
#define RVP_MAX_CSI_BYTES 512U
#define RVP_BUILD_ID_LEN 16U

typedef enum {
    RVP_ROLE_TX = 1,
    RVP_ROLE_RX = 2,
} rvp_role_t;

typedef enum {
    RVP_PACKET_CSI = 1,
    RVP_PACKET_STATUS = 2,
} rvp_packet_type_t;

enum {
    RVP_CSI_FLAG_FIRST_WORD_INVALID = 1U << 0,
    RVP_CSI_FLAG_SINK_VALID = 1U << 1,
    RVP_CSI_FLAG_PROBE_VALID = 1U << 2,
};

enum {
    RVP_STATUS_FLAG_GAIN_LOCKED = 1U << 0,
    RVP_STATUS_FLAG_SINK_VALID = 1U << 1,
    RVP_STATUS_FLAG_TX_PROBE_VALID = 1U << 2,
};

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint8_t version;
    uint8_t reserved;
    uint16_t probe_rate_hz;
    uint32_t sequence;
} rvp_probe_payload_t;

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint8_t version;
    uint8_t packet_type;
    uint8_t node_id;
    uint8_t role;
    uint32_t sequence;
    uint64_t timestamp_us;
    uint8_t tx_mac[6];
    uint8_t rx_mac[6];
    uint8_t channel;
    uint8_t bandwidth;
    int8_t rssi;
    int8_t noise_floor;
    int8_t fft_gain;
    uint8_t agc_gain;
    uint16_t csi_len;
    uint16_t flags;
    int16_t gain_compensation_q8;
    uint32_t uptime_ms;
} rvp_csi_packet_header_t;

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint8_t version;
    uint8_t packet_type;
    uint8_t node_id;
    uint8_t role;
    uint32_t uptime_ms;
    uint8_t tx_mac[6];
    uint8_t rx_mac[6];
    uint8_t channel;
    uint8_t bandwidth;
    uint16_t flags;
    uint32_t frames_received;
    uint32_t frames_sent;
    uint32_t frames_dropped;
    uint32_t last_sequence;
    char build_id[RVP_BUILD_ID_LEN];
    uint16_t probe_rate_hz;
    uint16_t reboot_count;
} rvp_status_packet_t;

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint8_t version;
    uint8_t reserved;
    uint16_t sink_port;
    uint32_t nonce;
    uint32_t ttl_ms;
} rvp_discovery_packet_t;

_Static_assert(sizeof(rvp_csi_packet_header_t) == 48,
               "CSI packet header must remain 48 bytes");
_Static_assert(sizeof(rvp_status_packet_t) == 64,
               "status packet must remain 64 bytes");
_Static_assert(sizeof(rvp_discovery_packet_t) == 16,
               "discovery packet must remain 16 bytes");
_Static_assert(sizeof(rvp_probe_payload_t) == 12,
               "probe payload must remain 12 bytes");

esp_err_t rvp_parse_mac(const char *text, uint8_t mac[6]);
esp_err_t rvp_identity_led_init(rvp_role_t role, uint8_t node_id);

#ifdef __cplusplus
}
#endif
