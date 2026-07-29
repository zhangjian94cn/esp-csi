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

execute_process(
    COMMAND git rev-parse --short=12 HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../../../.."
    OUTPUT_VARIABLE RVP_SOURCE_BUILD_ID
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT RVP_SOURCE_BUILD_ID)
    set(RVP_SOURCE_BUILD_ID "source-unknown")
endif()
target_compile_definitions(
    ${COMPONENT_LIB}
    PRIVATE RVP_SOURCE_BUILD_ID=\"${RVP_SOURCE_BUILD_ID}\"
)
