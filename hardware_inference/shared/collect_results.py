"""
Reads inference results printed over UART from a NUCLEO board
and saves them to a CSV file.

Firmware must print lines in this exact format:
    RESULT,<pipeline_name>,<cycles>,<latency_ms>,<arena_bytes>

And signal completion with:
    DONE

ROM bytes are not sent over serial (they are build-time constants).
Pass --rom <bytes> to record them alongside runtime measurements.

Usage:
    python collect_results.py \
        --port /dev/ttyACM0 \
        --baud 115200 \
        --output results/hardware/f401re_seed0.csv \
        --rom 98432 \
        --timeout 120
"""

import argparse
import csv
import os
import time

import serial


RESULT_PREFIX = "RESULT,"
DONE_MSG      = "DONE"


def collect(
    port: str,
    baud: int,
    output_path: str,
    rom_bytes: int,
    timeout_s: int,
) -> None:
    """
    Opens a serial connection to the board, collects RESULT lines,
    and writes them to a CSV. Exits on DONE signal or timeout.

    Args:
        port:        Serial port, e.g. /dev/ttyACM0
        baud:        Baud rate, must match firmware UART config
        output_path: CSV output path
        rom_bytes:   Flash used (text+data), from arm-none-eabi-size at build time
        timeout_s:   Seconds to wait before giving up if DONE is never received
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Opening {port} at {baud} baud...")

    with serial.Serial(port, baud, timeout=1) as ser:
        # Opening the port triggers a DTR reset on Nucleo boards.
        # Wait for the board to boot before reading.
        time.sleep(2)
        ser.reset_input_buffer()

        rows    = []
        deadline = time.time() + timeout_s
        timed_out = False

        print(f"Collecting results (timeout: {timeout_s}s). Press Ctrl+C to stop early.\n")

        try:
            while time.time() < deadline:
                line = ser.readline().decode("utf-8", errors="ignore").strip()

                if not line:
                    continue

                print(f"  Board: {line}")

                if line.startswith(RESULT_PREFIX):
                    _parse_result(line, rom_bytes, rows)

                elif line == DONE_MSG:
                    print("\nBoard signalled DONE.")
                    break

            else:
                # Loop exited via deadline, not via DONE
                timed_out = True
                print(f"\nTimeout reached ({timeout_s}s). Board did not send DONE.")

        except KeyboardInterrupt:
            print("\nStopped by user.")

    # Serial port is now closed (context manager handles it)

    if not rows:
        print("No results collected. CSV not written.")
        return

    _write_csv(output_path, rows)

    if timed_out:
        print("WARNING: results may be incomplete (timeout before DONE).")


def _parse_result(line: str, rom_bytes: int, rows: list) -> None:
    """
    Parses a RESULT line from firmware and appends to rows.

    Expected format:
        RESULT,<pipeline_name>,<cycles>,<latency_ms>,<arena_bytes>
    """
    parts = line[len(RESULT_PREFIX):].split(",")

    if len(parts) != 4:
        print(f"    WARNING: malformed RESULT line, expected 4 fields, got {len(parts)}: {line}")
        return

    try:
        row = {
            "pipeline":    parts[0].strip(),
            "cycles":      int(parts[1].strip()),
            "latency_ms":  float(parts[2].strip()),
            "arena_bytes": int(parts[3].strip()),
            "rom_bytes":   rom_bytes,  # from build time, not firmware
        }
    except ValueError as e:
        print(f"    WARNING: could not parse RESULT fields: {e}")
        return

    rows.append(row)
    print(f"    -> Recorded: {row}")


def _write_csv(output_path: str, rows: list) -> None:
    """Writes collected rows to CSV."""
    fieldnames = ["pipeline", "cycles", "latency_ms", "arena_bytes", "rom_bytes"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)