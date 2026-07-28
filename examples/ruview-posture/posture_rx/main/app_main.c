/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "nvs_flash.h"

#include "esp_csi_gain_ctrl.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_timer.h"
#include "esp_wifi.h"

#include "lwip/inet.h"
#include "lwip/sockets.h"

#include "protocol_examples_common.h"
#include "ruview_posture_protocol.h"

#define RVP_QUEUE_LENGTH 24U
#define RVP_SINK_STALE_MS 10000U

typedef struct {
    rvp_csi_packet_header_t header;
    uint8_t csi[RVP_MAX_CSI_BYTES];
} rvp_queued_csi_t;

static const char *TAG = "rvp_rx";
static QueueHandle_t s_csi_queue;
static uint8_t s_tx_mac[6];
static uint8_t s_rx_mac[6];
static uint8_t s_channel;
static struct sockaddr_in s_sink = {0};
static int64_t s_sink_expires_us;
static portMUX_TYPE s_sink_lock = portMUX_INITIALIZER_UNLOCKED;
static uint32_t s_frames_received;
static uint32_t s_frames_sent;
static uint32_t s_frames_dropped;
static uint32_t s_last_sequence;

static void init_nvs(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES ||
        err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
}

static bool current_sink(struct sockaddr_in *sink)
{
    bool valid = false;
    int64_t now = esp_timer_get_time();
    portENTER_CRITICAL(&s_sink_lock);
    if (s_sink.sin_family == AF_INET && now < s_sink_expires_us) {
        *sink = s_sink;
        valid = true;
    }
    portEXIT_CRITICAL(&s_sink_lock);
    return valid;
}

static void discovery_task(void *arg)
{
    (void)arg;
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "create discovery socket failed errno=%d", errno);
        vTaskDelete(NULL);
        return;
    }

    struct sockaddr_in bind_addr = {
        .sin_family = AF_INET,
        .sin_port = htons(RVP_DISCOVERY_PORT),
        .sin_addr.s_addr = htonl(INADDR_ANY),
    };
    if (bind(sock, (struct sockaddr *)&bind_addr, sizeof(bind_addr)) != 0) {
        ESP_LOGE(TAG, "bind discovery socket failed errno=%d", errno);
        close(sock);
        vTaskDelete(NULL);
        return;
    }

    while (true) {
        rvp_discovery_packet_t packet = {0};
        struct sockaddr_in source = {0};
        socklen_t source_len = sizeof(source);
        ssize_t received = recvfrom(
            sock,
            &packet,
            sizeof(packet),
            0,
            (struct sockaddr *)&source,
            &source_len);
        if (received != sizeof(packet) ||
            packet.magic != RVP_DISCOVERY_MAGIC ||
            packet.version != RVP_PROTOCOL_VERSION ||
            packet.sink_port == 0) {
            continue;
        }

        uint32_t ttl_ms = packet.ttl_ms;
        if (ttl_ms < 2000U || ttl_ms > 60000U) {
            ttl_ms = RVP_SINK_STALE_MS;
        }
        source.sin_port = htons(packet.sink_port);
        portENTER_CRITICAL(&s_sink_lock);
        s_sink = source;
        s_sink_expires_us =
            esp_timer_get_time() + ((int64_t)ttl_ms * 1000LL);
        portEXIT_CRITICAL(&s_sink_lock);
        ESP_LOGI(TAG, "sink discovered at %s:%u nonce=%" PRIu32,
                 inet_ntoa(source.sin_addr),
                 packet.sink_port,
                 packet.nonce);
    }
}

static void stream_task(void *arg)
{
    (void)arg;
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "create stream socket failed errno=%d", errno);
        vTaskDelete(NULL);
        return;
    }

    int64_t last_status_us = 0;
    rvp_queued_csi_t queued = {0};
    uint8_t datagram[sizeof(rvp_csi_packet_header_t) + RVP_MAX_CSI_BYTES];

    while (true) {
        struct sockaddr_in sink = {0};
        bool sink_valid = current_sink(&sink);
        if (xQueueReceive(s_csi_queue, &queued, pdMS_TO_TICKS(50)) == pdTRUE &&
            sink_valid) {
            queued.header.flags |= RVP_CSI_FLAG_SINK_VALID;
            memcpy(datagram, &queued.header, sizeof(queued.header));
            memcpy(datagram + sizeof(queued.header),
                   queued.csi,
                   queued.header.csi_len);
            size_t length = sizeof(queued.header) + queued.header.csi_len;
            if (sendto(sock,
                       datagram,
                       length,
                       0,
                       (struct sockaddr *)&sink,
                       sizeof(sink)) == (ssize_t)length) {
                s_frames_sent++;
            } else {
                s_frames_dropped++;
            }
        }

        int64_t now = esp_timer_get_time();
        if (sink_valid && now - last_status_us >= 1000000LL) {
            rvp_status_packet_t status = {
                .magic = RVP_STATUS_MAGIC,
                .version = RVP_PROTOCOL_VERSION,
                .packet_type = RVP_PACKET_STATUS,
                .node_id = CONFIG_RVP_NODE_ID,
                .role = RVP_ROLE_RX,
                .uptime_ms = (uint32_t)(now / 1000LL),
                .channel = s_channel,
                .bandwidth = 20,
                .frames_received = s_frames_received,
                .frames_sent = s_frames_sent,
                .frames_dropped = s_frames_dropped,
                .last_sequence = s_last_sequence,
            };
            memcpy(status.tx_mac, s_tx_mac, sizeof(status.tx_mac));
            memcpy(status.rx_mac, s_rx_mac, sizeof(status.rx_mac));
            snprintf(status.build_id, sizeof(status.build_id), "%.15s", "8633d671-rvp1");
            (void)sendto(sock,
                         &status,
                         sizeof(status),
                         0,
                         (struct sockaddr *)&sink,
                         sizeof(sink));
            last_status_us = now;
        }
    }
}

static void csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    (void)ctx;
    if (!info || !info->buf || memcmp(info->mac, s_tx_mac, 6) != 0) {
        return;
    }

    rvp_queued_csi_t queued = {0};
    size_t csi_len = info->len;
    if (csi_len > RVP_MAX_CSI_BYTES) {
        csi_len = RVP_MAX_CSI_BYTES;
    }

    uint32_t sequence = s_frames_received;
    if (info->payload && info->rx_ctrl.sig_len >= 19U) {
        memcpy(&sequence, info->payload + 15, sizeof(sequence));
    }

    uint8_t agc_gain = 0;
    int8_t fft_gain = 0;
    float gain_compensation = 1.0f;
    esp_csi_gain_ctrl_get_rx_gain(&info->rx_ctrl, &agc_gain, &fft_gain);
    esp_csi_gain_ctrl_get_gain_compensation(
        &gain_compensation,
        agc_gain,
        fft_gain);

    queued.header.magic = RVP_CSI_MAGIC;
    queued.header.version = RVP_PROTOCOL_VERSION;
    queued.header.packet_type = RVP_PACKET_CSI;
    queued.header.node_id = CONFIG_RVP_NODE_ID;
    queued.header.role = RVP_ROLE_RX;
    queued.header.sequence = sequence;
    queued.header.timestamp_us = (uint64_t)esp_timer_get_time();
    memcpy(queued.header.tx_mac, s_tx_mac, sizeof(s_tx_mac));
    memcpy(queued.header.rx_mac, s_rx_mac, sizeof(s_rx_mac));
    queued.header.channel = info->rx_ctrl.channel;
    queued.header.bandwidth = info->rx_ctrl.cwb ? 40U : 20U;
    queued.header.rssi = info->rx_ctrl.rssi;
    queued.header.noise_floor = info->rx_ctrl.noise_floor;
    queued.header.fft_gain = fft_gain;
    queued.header.agc_gain = agc_gain;
    queued.header.csi_len = (uint16_t)csi_len;
    queued.header.flags =
        info->first_word_invalid ? RVP_CSI_FLAG_FIRST_WORD_INVALID : 0U;
    float gain_q8 = gain_compensation * 256.0f;
    if (gain_q8 > INT16_MAX) {
        gain_q8 = INT16_MAX;
    } else if (gain_q8 < INT16_MIN) {
        gain_q8 = INT16_MIN;
    }
    queued.header.gain_compensation_q8 = (int16_t)lroundf(gain_q8);
    queued.header.uptime_ms =
        (uint32_t)(queued.header.timestamp_us / 1000ULL);
    memcpy(queued.csi, info->buf, csi_len);

    s_frames_received++;
    s_last_sequence = sequence;
    if (xQueueSend(s_csi_queue, &queued, 0) != pdTRUE) {
        s_frames_dropped++;
    }
}

static void init_esp_now_and_csi(void)
{
    ESP_ERROR_CHECK(esp_now_init());

    esp_now_peer_info_t peer = {
        .channel = s_channel,
        .ifidx = WIFI_IF_STA,
        .encrypt = false,
    };
    memset(peer.peer_addr, 0xff, sizeof(peer.peer_addr));
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));

    esp_now_rate_config_t rate = {
        .phymode = WIFI_PHY_MODE_HT20,
        .rate = WIFI_PHY_RATE_MCS0_LGI,
        .ersu = false,
        .dcm = false,
    };
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer.peer_addr, &rate));

    wifi_csi_config_t config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = true,
        .manu_scale = false,
        .shift = false,
    };
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

void app_main(void)
{
    init_nvs();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(example_connect());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20));

    wifi_ap_record_t ap = {0};
    ESP_ERROR_CHECK(esp_wifi_sta_get_ap_info(&ap));
    s_channel = ap.primary;

    ESP_ERROR_CHECK(rvp_parse_mac(CONFIG_RVP_TX_MAC, s_tx_mac));
    ESP_ERROR_CHECK(esp_read_mac(s_rx_mac, ESP_MAC_WIFI_STA));
    s_csi_queue = xQueueCreate(RVP_QUEUE_LENGTH, sizeof(rvp_queued_csi_t));
    ESP_ERROR_CHECK(s_csi_queue ? ESP_OK : ESP_ERR_NO_MEM);

    init_esp_now_and_csi();
    ESP_ERROR_CHECK(rvp_identity_led_init(RVP_ROLE_RX, CONFIG_RVP_NODE_ID));

    BaseType_t discovery_ok = xTaskCreate(
        discovery_task, "rvp_discovery", 4096, NULL, 5, NULL);
    BaseType_t stream_ok = xTaskCreate(
        stream_task, "rvp_stream", 6144, NULL, 6, NULL);
    ESP_ERROR_CHECK(
        discovery_ok == pdPASS && stream_ok == pdPASS
            ? ESP_OK
            : ESP_ERR_NO_MEM);

    ESP_LOGI(TAG,
             "ready node=%u rx=" MACSTR " tx=" MACSTR
             " channel=%u data_port=%u discovery_port=%u",
             CONFIG_RVP_NODE_ID,
             MAC2STR(s_rx_mac),
             MAC2STR(s_tx_mac),
             s_channel,
             RVP_CSI_PORT,
             RVP_DISCOVERY_PORT);
}
