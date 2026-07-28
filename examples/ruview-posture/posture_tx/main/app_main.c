/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include <inttypes.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"

#include "esp_event.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_wifi.h"

#include "protocol_examples_common.h"
#include "ruview_posture_protocol.h"

#define RVP_SEND_FREQUENCY_HZ 50U

static const char *TAG = "rvp_tx";
static const uint8_t s_broadcast_mac[6] = {
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};

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

static void init_esp_now(uint8_t channel)
{
    ESP_ERROR_CHECK(esp_now_init());

    esp_now_peer_info_t peer = {
        .channel = channel,
        .ifidx = WIFI_IF_STA,
        .encrypt = false,
    };
    memcpy(peer.peer_addr, s_broadcast_mac, sizeof(peer.peer_addr));
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));

    esp_now_rate_config_t rate = {
        .phymode = WIFI_PHY_MODE_HT20,
        .rate = WIFI_PHY_RATE_MCS0_LGI,
        .ersu = false,
        .dcm = false,
    };
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer.peer_addr, &rate));
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

    uint8_t own_mac[6] = {0};
    ESP_ERROR_CHECK(esp_read_mac(own_mac, ESP_MAC_WIFI_STA));
    init_esp_now(ap.primary);
    ESP_ERROR_CHECK(rvp_identity_led_init(RVP_ROLE_TX, CONFIG_RVP_NODE_ID));

    ESP_LOGI(TAG,
             "ready node=%u mac=" MACSTR " channel=%u bandwidth=HT20 rate=%uHz",
             CONFIG_RVP_NODE_ID,
             MAC2STR(own_mac),
             ap.primary,
             RVP_SEND_FREQUENCY_HZ);

    const TickType_t interval = pdMS_TO_TICKS(1000U / RVP_SEND_FREQUENCY_HZ);
    TickType_t next_wake = xTaskGetTickCount();
    uint32_t sequence = 0;
    while (true) {
        esp_err_t err = esp_now_send(
            s_broadcast_mac,
            (const uint8_t *)&sequence,
            sizeof(sequence));
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "send sequence=%" PRIu32 " failed: %s",
                     sequence, esp_err_to_name(err));
        }
        sequence++;
        vTaskDelayUntil(&next_wake, interval);
    }
}
