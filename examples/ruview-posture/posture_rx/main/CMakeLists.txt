idf_component_register(
    SRCS "app_main.c"
    INCLUDE_DIRS "."
    REQUIRES
        esp_csi_gain_ctrl
        esp_event
        esp_netif
        esp_timer
        esp_wifi
        lwip
        nvs_flash
        protocol_examples_common
        ruview_posture_common
)
