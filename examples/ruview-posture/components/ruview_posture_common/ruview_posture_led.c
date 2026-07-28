/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ruview_posture_protocol.h"

#include <stdio.h>

#include "esp_check.h"
#include "esp_log.h"
#include "led_strip.h"
#include "sdkconfig.h"

static const char *TAG = "rvp_common";

esp_err_t rvp_parse_mac(const char *text, uint8_t mac[6])
{
    if (!text || !mac) {
        return ESP_ERR_INVALID_ARG;
    }

    unsigned int parsed[6] = {0};
    if (sscanf(text, "%02x:%02x:%02x:%02x:%02x:%02x",
               &parsed[0], &parsed[1], &parsed[2],
               &parsed[3], &parsed[4], &parsed[5]) != 6) {
        return ESP_ERR_INVALID_ARG;
    }

    for (size_t i = 0; i < 6; ++i) {
        if (parsed[i] > UINT8_MAX) {
            return ESP_ERR_INVALID_ARG;
        }
        mac[i] = (uint8_t)parsed[i];
    }
    return ESP_OK;
}

esp_err_t rvp_identity_led_init(rvp_role_t role, uint8_t node_id)
{
#if !CONFIG_RVP_LED_ENABLE
    (void)role;
    (void)node_id;
    return ESP_OK;
#else
    led_strip_config_t strip_config = {
        .strip_gpio_num = CONFIG_RVP_LED_GPIO,
        .max_leds = 1,
    };
    led_strip_rmt_config_t rmt_config = {
        .resolution_hz = 10 * 1000 * 1000,
        .flags.with_dma = false,
    };
    led_strip_handle_t strip = NULL;
    ESP_RETURN_ON_ERROR(
        led_strip_new_rmt_device(&strip_config, &rmt_config, &strip),
        TAG,
        "create identity LED failed");

    uint8_t red = 0;
    uint8_t green = 0;
    uint8_t blue = 0;
    if (role == RVP_ROLE_TX || node_id == 1) {
        red = 48;
    } else if (node_id == 2) {
        green = 48;
    } else {
        blue = 48;
    }

    ESP_RETURN_ON_ERROR(
        led_strip_set_pixel(strip, 0, red, green, blue),
        TAG,
        "set identity LED failed");
    ESP_RETURN_ON_ERROR(
        led_strip_refresh(strip),
        TAG,
        "refresh identity LED failed");
    ESP_LOGI(TAG, "identity LED node=%u rgb=(%u,%u,%u)",
             node_id, red, green, blue);
    return ESP_OK;
#endif
}
