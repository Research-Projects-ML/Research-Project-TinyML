import os
import time
import edgeimpulse as ei


EI_DEVICE_MAP = {
    'cortex_m4f': 'cortex-m4f-80mhz',
    'cortex_m7':  'cortex-m7-216mhz',
}


def _parse_variant(variant, tflite_path):
    if variant is None:
        return None

    try:
        tflite_mem = variant.memory.tflite
        eon_mem    = getattr(variant.memory, 'eon', None)

        ram_bytes        = tflite_mem.ram
        rom_bytes        = tflite_mem.rom
        arena_size_bytes = getattr(tflite_mem, 'arena_size', None)

        eon_ram_bytes = getattr(eon_mem, 'ram', None) if eon_mem else None
        eon_rom_bytes = getattr(eon_mem, 'rom', None) if eon_mem else None

        latency_ms  = getattr(variant, 'time_per_inference_ms', None)
        is_supported = getattr(variant, 'is_supported_on_mcu',
                        getattr(variant, 'isSupportedOnMcu', None))

        tflite_file_size_bytes = os.path.getsize(tflite_path)

        return {
            'ram_bytes':            ram_bytes,
            'rom_bytes':            rom_bytes,
            'arena_size_bytes':     arena_size_bytes,
            'eon_ram_bytes':        eon_ram_bytes,
            'eon_rom_bytes':        eon_rom_bytes,
            'latency_ms':           latency_ms,
            'tflite_file_size_bytes': tflite_file_size_bytes,
            'is_supported':         bool(is_supported) if is_supported is not None else None,
            'ram_kb':               round(ram_bytes / 1024, 3),
            'rom_kb':               round(rom_bytes / 1024, 3),
            'arena_size_kb':        round(arena_size_bytes / 1024, 3) if arena_size_bytes else None,
            'eon_ram_kb':           round(eon_ram_bytes / 1024, 3) if eon_ram_bytes else None,
            'eon_rom_kb':           round(eon_rom_bytes / 1024, 3) if eon_rom_bytes else None,
            'tflite_file_size_kb':  round(tflite_file_size_bytes / 1024, 3),
        }

    except Exception as e:
        print(f"    [EI] Parse error: {e}")
        return None


def _get_variant(resp):
    try:
        info    = resp.model.profile_info
        variant = getattr(info, 'int8', None) or getattr(info, 'float32', None)
        return variant
    except Exception as e:
        print(f"    [EI] Could not extract variant from response: {e}")
        return None


def _empty_profile(reason='failed'):
    return {
        'ram_bytes':              None,
        'rom_bytes':              None,
        'arena_size_bytes':       None,
        'eon_ram_bytes':          None,
        'eon_rom_bytes':          None,
        'latency_ms':             None,
        'tflite_file_size_bytes': None,
        'is_supported':           False,
        'ram_kb':                 None,
        'rom_kb':                 None,
        'arena_size_kb':          None,
        'eon_ram_kb':             None,
        'eon_rom_kb':             None,
        'tflite_file_size_kb':    None,
        'error':                  reason,
    }


def _profile_m0plus(tflite_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"  [EI] Profiling cortex_m0plus (generic low-end estimate)...")
            resp    = ei.model.profile(model=tflite_path)
            variant = _get_variant(resp)
            parsed  = _parse_variant(variant, tflite_path)

            if parsed:
                print(
                    f"     RAM: {parsed['ram_kb']} KB | "
                    f"ROM: {parsed['rom_kb']} KB | "
                    f"Arena: {parsed['arena_size_kb']} KB | "
                    f"EON RAM: {parsed['eon_ram_kb']} KB | "
                    f"EON ROM: {parsed['eon_rom_kb']} KB | "
                    f"Latency: {parsed['latency_ms']} ms | "
                    f"File: {parsed['tflite_file_size_kb']} KB | "
                    f"Supported: {parsed['is_supported']}"
                )
                return parsed
            else:
                raise ValueError("Parsed result was None")

        except Exception as e:
            print(f"  [EI] M0+ attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    return _empty_profile('m0plus_profiling_failed')


def _profile_device(tflite_path, mcu_name, device, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"  [EI] Profiling {mcu_name} ({device})...")
            resp    = ei.model.profile(model=tflite_path, device=device)
            variant = _get_variant(resp)
            parsed  = _parse_variant(variant, tflite_path)

            if parsed:
                print(
                    f"     RAM: {parsed['ram_kb']} KB | "
                    f"ROM: {parsed['rom_kb']} KB | "
                    f"Arena: {parsed['arena_size_kb']} KB | "
                    f"EON RAM: {parsed['eon_ram_kb']} KB | "
                    f"EON ROM: {parsed['eon_rom_kb']} KB | "
                    f"Latency: {parsed['latency_ms']} ms | "
                    f"File: {parsed['tflite_file_size_kb']} KB | "
                    f"Supported: {parsed['is_supported']}"
                )
                return parsed
            else:
                raise ValueError("Parsed result was None")

        except Exception as e:
            print(f"  [EI] {mcu_name} attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    return _empty_profile(f'{mcu_name}_profiling_failed')


def profile_tflite(tflite_path, max_retries=3):
    results = {}

    results['cortex_m0plus'] = _profile_m0plus(tflite_path, max_retries)

    for mcu_name, device in EI_DEVICE_MAP.items():
        results[mcu_name] = _profile_device(
            tflite_path, mcu_name, device, max_retries
        )

    return results


def assess_deployability(hardware_profile, hardware_config):
    results = {}

    for mcu_name, budget in hardware_config.items():
        profile = hardware_profile.get(mcu_name, {})

        ram_kb = profile.get('ram_kb')
        rom_kb = profile.get('rom_kb')

        ram_ok = ram_kb is not None and ram_kb <= budget['ram_kb']
        rom_ok = rom_kb is not None and rom_kb <= budget['flash_kb']

        results[mcu_name] = {
            'deployable':    ram_ok and rom_ok,
            'ram_ok':        ram_ok,
            'rom_ok':        rom_ok,
            'ram_used_kb':   ram_kb,
            'rom_used_kb':   rom_kb,
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

        rom_kb = None
        for mcu_data in r['hardware_profile'].values():
            if mcu_data.get('rom_kb') is not None:
                rom_kb = mcu_data['rom_kb']
                break

        if rom_kb is None:
            continue

        points.append({
            'pipeline': r['pipeline'],
            'domain':   r['domain'],
            'seed':     r['seed'],
            'accuracy': acc,
            'rom_kb':   rom_kb,
        })

    pareto = []
    for candidate in points:
        dominated = any(
            other['accuracy'] >= candidate['accuracy']
            and other['rom_kb'] <= candidate['rom_kb']
            and (
                other['accuracy'] > candidate['accuracy']
                or other['rom_kb'] < candidate['rom_kb']
            )
            for other in points
            if other is not candidate
        )
        if not dominated:
            pareto.append(candidate)

    pareto.sort(key=lambda x: x['rom_kb'])
    return pareto