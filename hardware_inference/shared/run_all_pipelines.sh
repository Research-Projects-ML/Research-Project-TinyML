#!/usr/bin/env bash
# Rebuilds firmware for each pipeline, flashes it, and collects results over serial.
#
# Usage:
#   ./run_all_pipelines.sh --board f401re --seed 0 --port /dev/ttyACM0
#
# Assumes:
#   - tflite files at: results/pipeline_results/timeseries/seed_<N>/<pipeline>_final.tflite
#   - firmware source at: hardware_inference/nucleo_<board>/
#   - shared scripts at:  hardware_inference/shared/

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
BOARD="f401re"
SEED=0
PORT="/dev/ttyACM0"
BAUD=115200
TIMEOUT=120

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --board)   BOARD="$2";   shift 2 ;;
        --seed)    SEED="$2";    shift 2 ;;
        --port)    PORT="$2";    shift 2 ;;
        --baud)    BAUD="$2";    shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SHARED_DIR="$REPO_ROOT/hardware_inference/shared"
BOARD_DIR="$REPO_ROOT/hardware_inference/nucleo_${BOARD}"
BUILD_DIR="$BOARD_DIR/build"
TFLITE_DIR="$REPO_ROOT/results/pipeline_results/timeseries/seed_${SEED}"
OUTPUT_CSV="$REPO_ROOT/results/hardware/${BOARD}_seed${SEED}.csv"

PIPELINES=(
    P_KD_QAT
    P_QAT_KD
    KD_P_QAT
    KD_QAT_P
    QAT_P_KD
    QAT_KD_P
    P_only
    KD_only
    QAT_only
    baseline
)

# ── Build directory setup (once) ─────────────────────────────────────────────
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -DCMAKE_TOOLCHAIN_FILE=../arm-none-eabi.cmake -DCMAKE_BUILD_TYPE=Release
cd "$REPO_ROOT"

# ── Per-pipeline loop ─────────────────────────────────────────────────────────
for PIPELINE in "${PIPELINES[@]}"; do
    echo ""
    echo "══════════════════════════════════════════════"
    echo "  Pipeline : $PIPELINE  |  Seed : $SEED"
    echo "══════════════════════════════════════════════"

    TFLITE_PATH="$TFLITE_DIR/${PIPELINE}_final.tflite"
    MODEL_CC="$BOARD_DIR/src/model_data.cc"

    # 1. Convert .tflite → model_data.cc + model_data.h
    if [[ ! -f "$TFLITE_PATH" ]]; then
        echo "  SKIP: tflite not found at $TFLITE_PATH"
        continue
    fi

    python "$SHARED_DIR/convert_to_c_array.py" \
        --tflite "$TFLITE_PATH" \
        --output "$MODEL_CC"

    # 2. Rebuild firmware (only model_data.cc changes, so this is fast)
    cd "$BUILD_DIR"
    make -j"$(nproc)"
    ELF_PATH="$BUILD_DIR/nucleo_${BOARD}.elf"

    # 3. Extract ROM from ELF (text + data sections)
    ROM_BYTES=$(arm-none-eabi-size "$ELF_PATH" | awk 'NR==2 {print $1+$2}')
    echo "  ROM: $ROM_BYTES bytes"

    cd "$REPO_ROOT"

    # 4. Flash firmware
    STM32_Programmer_CLI \
        -c port=SWD \
        -w "$BUILD_DIR/nucleo_${BOARD}.bin" 0x08000000 \
        -v -rst

    # 5. Wait for board to boot
    sleep 3

    # 6. Collect results over serial, appending ROM from build step
    python "$SHARED_DIR/collect_results.py" \
        --port    "$PORT" \
        --baud    "$BAUD" \
        --output  "$OUTPUT_CSV" \
        --rom     "$ROM_BYTES" \
        --timeout "$TIMEOUT"

done

echo ""
echo "All pipelines done. Results at: $OUTPUT_CSV"