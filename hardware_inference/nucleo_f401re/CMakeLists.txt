cmake_minimum_required(VERSION 3.16)
project(nucleo_f401re C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Pipeline name ─────────────────────────────────────────────────────────────
if(NOT DEFINED PIPELINE_NAME)
    set(PIPELINE_NAME "unknown")
endif()
add_compile_definitions(PIPELINE_NAME="${PIPELINE_NAME}")

# ── Paths ─────────────────────────────────────────────────────────────────────
set(TFLM_ROOT       "$ENV{HOME}/tflm_m4f_ref")
set(HAL_ROOT        "$ENV{HOME}/Downloads/STM32CubeF4/Drivers/STM32F4xx_HAL_Driver")
set(CMSIS_INC       "$ENV{HOME}/Downloads/STM32CubeF4/Drivers/CMSIS/Include")
set(CMSIS_DEV_ROOT  "$ENV{HOME}/Downloads/cmsis-device-f4")
set(LINKER_SCRIPT   "$ENV{HOME}/Downloads/STM32CubeF4/Projects/STM32F401RE-Nucleo/Templates/STM32CubeIDE/STM32F401RETX_FLASH.ld")

# ── TFLM sources: micro/ and supporting directories ONLY ─────────────────────
# We do NOT glob the whole tree. We include only the specific subdirectories
# that belong in an MCU build, explicitly excluding desktop TFLite files.

file(GLOB_RECURSE TFLM_MICRO_SRCS
    "${TFLM_ROOT}/tensorflow/lite/micro/*.cc"
)
file(GLOB_RECURSE TFLM_SCHEMA_SRCS
    "${TFLM_ROOT}/tensorflow/compiler/mlir/lite/schema/schema_utils.cc"
)
# signal not needed for TCN model
# # signal/ excluded — audio DSP library, not needed for TCN timeseries model
file(GLOB_RECURSE TFLM_THIRD_PARTY_SRCS
    "${TFLM_ROOT}/third_party/*.cc"
)

# Combine
set(TFLM_SOURCES
    ${TFLM_MICRO_SRCS}
    ${TFLM_SCHEMA_SRCS}

    ${TFLM_THIRD_PARTY_SRCS}
)

# Exclude test files and TFLM's own micro_time (we provide our own)
list(FILTER TFLM_SOURCES EXCLUDE REGEX ".*_test\\.cc$")
list(FILTER TFLM_SOURCES EXCLUDE REGEX ".*/cortex_m_generic/micro_time\\.cc$")

# ── HAL sources ───────────────────────────────────────────────────────────────
file(GLOB HAL_SOURCES "${HAL_ROOT}/Src/stm32f4xx_hal*.c")
list(FILTER HAL_SOURCES EXCLUDE REGEX ".*_template\\.c$")

# ── Firmware sources ──────────────────────────────────────────────────────────
# These desktop TFLite files provide symbols that TFLM micro kernels depend on.
# They must be compiled with ARM flags, so we add them explicitly here
# rather than globbing the full desktop tree.
set(TFLM_SUPPORT_SRCS
    ${TFLM_ROOT}/tensorflow/lite/core/c/common.cc
    ${TFLM_ROOT}/tensorflow/lite/core/api/flatbuffer_conversions.cc
    ${TFLM_ROOT}/tensorflow/lite/core/api/tensor_utils.cc
    ${TFLM_ROOT}/tensorflow/lite/kernels/internal/quantization_util.cc
    ${TFLM_ROOT}/tensorflow/lite/kernels/internal/tensor_utils.cc
    ${TFLM_ROOT}/tensorflow/lite/kernels/internal/portable_tensor_utils.cc
    ${TFLM_ROOT}/tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cc
    ${TFLM_ROOT}/tensorflow/lite/kernels/internal/common.cc
    ${TFLM_ROOT}/tensorflow/lite/kernels/internal/tensor_ctypes.cc
    ${TFLM_ROOT}/tensorflow/lite/kernels/internal/runtime_shape.cc
    ${TFLM_ROOT}/tensorflow/lite/kernels/kernel_util.cc
    ${TFLM_ROOT}/tensorflow/compiler/mlir/lite/core/api/error_reporter.cc
)

set(FW_SOURCES
    src/main.cpp
    src/system_setup.cpp
    src/micro_time.cpp
    src/model_data.cc
    ${CMSIS_DEV_ROOT}/Source/Templates/gcc/startup_stm32f401xe.s
    ${CMSIS_DEV_ROOT}/Source/Templates/system_stm32f4xx.c
)

add_executable(nucleo_f401re
    ${FW_SOURCES}
    ${HAL_SOURCES}
    ${TFLM_SOURCES}
    ${TFLM_SUPPORT_SRCS}
)

# ── Include paths ─────────────────────────────────────────────────────────────
target_include_directories(nucleo_f401re PRIVATE
    src/
    ${TFLM_ROOT}
    ${TFLM_ROOT}/third_party/flatbuffers/include
    ${TFLM_ROOT}/third_party/gemmlowp
    ${TFLM_ROOT}/third_party/ruy
    ${HAL_ROOT}/Inc
    ${CMSIS_INC}
    ${CMSIS_DEV_ROOT}/Include
)

# ── Compiler flags ────────────────────────────────────────────────────────────
target_compile_options(nucleo_f401re PRIVATE
    $<$<COMPILE_LANGUAGE:C>:
        -mcpu=cortex-m4
        -mthumb
        -mfpu=fpv4-sp-d16
        -mfloat-abi=hard
        -Os
        -DSTM32F401xE
        -DUSE_HAL_DRIVER
        -DTF_LITE_STATIC_MEMORY
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -mcpu=cortex-m4
        -mthumb
        -mfpu=fpv4-sp-d16
        -mfloat-abi=hard
        -std=c++14
        -fno-rtti
        -fno-exceptions
        -fno-threadsafe-statics
        -fno-use-cxa-atexit
        -Os
        -DSTM32F401xE
        -DUSE_HAL_DRIVER
        -DTF_LITE_STATIC_MEMORY
    >
    $<$<COMPILE_LANGUAGE:ASM>:
        -mcpu=cortex-m4
        -mthumb
        -mfpu=fpv4-sp-d16
        -mfloat-abi=hard
    >
)

# ── Linker ────────────────────────────────────────────────────────────────────
target_link_options(nucleo_f401re PRIVATE
    -mcpu=cortex-m4
    -mthumb
    -mfpu=fpv4-sp-d16
    -mfloat-abi=hard
    -T${LINKER_SCRIPT}
    -Wl,--gc-sections
    -Wl,-Map=nucleo_f401re.map
    -specs=nano.specs
    -specs=nosys.specs
    -lm
)

# ── Post-build: generate .bin ─────────────────────────────────────────────────
add_custom_command(TARGET nucleo_f401re POST_BUILD
    COMMAND arm-none-eabi-objcopy -O binary
            $<TARGET_FILE:nucleo_f401re>
            ${CMAKE_CURRENT_BINARY_DIR}/nucleo_f401re.bin
    COMMENT "Generating nucleo_f401re.bin"
)
