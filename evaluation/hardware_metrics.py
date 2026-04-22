import os
import time
import numpy as np
import edgeimpulse as ei


EI_DEVICE_MAP = {
    'cortex_m0plus': None,
    'cortex_m4f':    'cortex-m4f-80mhz',
    'cortex_m7':     'cortex-m7-216mhz',
}


def profile_tflite_with_ei(tflite_path, hardware_config, max_retries=3):
    results = {}

    for mcu_name in EI_DEVICE_MAP:
        if mcu_name not in hardware_config:
            print(f"  [EI] {mcu_name} not in hardware_config, skipping.")
            continue

        device = EI_DEVICE_MAP[mcu_name]

        if device is None:
            results[mcu_name] = _profile_generic(tflite_path, max_retries)
        else:
            results[mcu_name] = _profile_device(tflite_path, device, max_retries)

    return results


def _profile_device(tflite_path, device, max_retries):
    for attempt in range(max_retries):
        try:
            print(f"  [EI] Profiling on {device}...")
            result = ei.model.profile(model=tflite_path, device=device)

            tflite_mem = result.model.profile_info.int8.memory.tflite
            return {
                'device':            device,
                'ram_bytes':         tflite_mem.ram,
                'rom_bytes':         tflite_mem.rom,
                'arena_size_bytes':  tflite_mem.arena_size,
                'inference_time_ms': result.model.profile_info.int8.time_per_inference_ms,
                'tflite_size_bytes': os.path.getsize(tflite_path),
                'is_supported':      True,
            }

        except Exception as e:
            print(f"  [EI] Attempt {attempt+1}/{max_retries} failed for {device}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    return _failed_profile(device)


def _profile_generic(tflite_path, max_retries):
    for attempt in range(max_retries):
        try:
            print(f"  [EI] Profiling on generic low-end MCU (Cortex-M0+ estimate)...")
            result = ei.model.profile(model=tflite_path)

            low_end = result.model.profile_info.int8.low_end_mcu
            return {
                'device':            'cortex_m0plus_generic_40mhz',
                'ram_bytes':         low_end.memory.tflite.ram,
                'rom_bytes':         low_end.memory.tflite.rom,
                'inference_time_ms': low_end.time_per_inference_ms,
                'tflite_size_bytes': os.path.getsize(tflite_path),
                'is_supported':      low_end.supported,
            }

        except Exception as e:
            print(f"  [EI] Attempt {attempt+1}/{max_retries} failed for M0+: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    return _failed_profile('cortex_m0plus_generic_40mhz')


def _failed_profile(device):
    return {
        'device':            device,
        'ram_bytes':         None,
        'rom_bytes':         None,
        'arena_size_bytes':  None,
        'inference_time_ms': None,
        'tflite_size_bytes': None,
        'is_supported':      False,
        'error':             'profiling_failed_after_retries',
    }


def assess_deployability(hardware_profile, hardware_config):
    results = {}

    for mcu_name, budget in hardware_config.items():
        profile = hardware_profile.get(mcu_name, {})

        ram_bytes = profile.get('ram_bytes')
        rom_bytes = profile.get('rom_bytes')

        ram_ok = ram_bytes is not None and (ram_bytes / 1024 <= budget['ram_kb'])
        rom_ok = rom_bytes is not None and (rom_bytes / 1024 <= budget['flash_kb'])

        results[mcu_name] = {
            'deployable':    ram_ok and rom_ok,
            'ram_ok':        ram_ok,
            'rom_ok':        rom_ok,
            'ram_used_kb':   round(ram_bytes / 1024, 2) if ram_bytes is not None else None,
            'rom_used_kb':   round(rom_bytes / 1024, 2) if rom_bytes is not None else None,
            'ram_budget_kb': budget['ram_kb'],
            'rom_budget_kb': budget['flash_kb'],
        }

    return results


def compute_pareto_frontier(results_list):
    candidates = [
        r for r in results_list
        if r.get('tflite_metrics') and r.get('hardware_profile')
    ]

    points = []
    for r in candidates:
        acc = r['tflite_metrics']['accuracy']

        rom = None
        for mcu_data in r['hardware_profile'].values():
            if mcu_data.get('rom_bytes') is not None:
                rom = mcu_data['rom_bytes']
                break

        if rom is None:
            continue

        points.append({
            'pipeline': r['pipeline'],
            'domain':   r['domain'],
            'seed':     r['seed'],
            'accuracy': acc,
            'rom_bytes': rom,
        })

    pareto = []
    for candidate in points:
        dominated = any(
            other['accuracy'] >= candidate['accuracy']
            and other['rom_bytes'] <= candidate['rom_bytes']
            and (
                other['accuracy'] > candidate['accuracy']
                or other['rom_bytes'] < candidate['rom_bytes']
            )
            for other in points
            if other is not candidate
        )
        if not dominated:
            pareto.append(candidate)

    pareto.sort(key=lambda x: x['rom_bytes'])
    return pareto