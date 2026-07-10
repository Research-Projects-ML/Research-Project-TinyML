import os
import time
import edgeimpulse as ei
from experiments.config import HARDWARE_CONFIG

# m4f and m7 have specific named device strings for EI profiling; 
# m0+ uses a generic call without a device string
EI_DEVICE_MAP = {
    target: cfg['ei_device']
    for target, cfg in HARDWARE_CONFIG.items()
    if cfg['ei_device'] is not None
}


def _parse_variant(variant, tflite_path):
    """
    Parses a named-device variant object (profile_info.int8 or float32).
    """
    if variant is None:
        return None

    try:
        tflite_mem = variant.memory.tflite
        eon_mem = getattr(variant.memory, 'eon', None)
        ram_bytes = tflite_mem.ram
        rom_bytes = tflite_mem.rom
        arena_size_bytes = getattr(tflite_mem, 'arena_size', None)
        eon_ram_bytes = getattr(eon_mem, 'ram', None) if eon_mem else None
        eon_rom_bytes = getattr(eon_mem, 'rom', None) if eon_mem else None
        latency_ms = getattr(variant, 'time_per_inference_ms', None)
        is_supported = getattr(variant, 'is_supported_on_mcu', getattr(variant, 'isSupportedOnMcu', None))

        return {
            'ram_bytes': ram_bytes,'rom_bytes': rom_bytes,'arena_size_bytes': arena_size_bytes,'eon_ram_bytes': eon_ram_bytes,'eon_rom_bytes': eon_rom_bytes,
            'latency_ms': latency_ms,'tflite_file_size_bytes': os.path.getsize(tflite_path),'is_supported': bool(is_supported) if is_supported is not None else None,
            'ram_kb': round(ram_bytes / 1024, 3),'rom_kb': round(rom_bytes / 1024, 3),
            'arena_size_kb': round(arena_size_bytes / 1024, 3) if arena_size_bytes else None,
            'eon_ram_kb': round(eon_ram_bytes / 1024, 3) if eon_ram_bytes else None,
            'eon_rom_kb': round(eon_rom_bytes / 1024, 3) if eon_rom_bytes else None,
            'tflite_file_size_kb': round(os.path.getsize(tflite_path) / 1024, 3),
        }
    except Exception as e:
        print(f"[EI] Parse error: {e}")
        return None


def _parse_low_end_mcu(low_end, tflite_path):
    """
    Parses the low-end MCU variant object from a generic EI profile call for M0+.
    time_per_inference_ms (latency at EI fixed 40 MHz reference)
    memory.tflite.ram (tensor arena RAM in bytes)
    memory.tflite.rom (model ROM in bytes)
    is_supported_on_mcu (operator support flag)
    variant (device category label).
    """
    if low_end is None:
        return None

    try:
        tflite_mem = low_end.memory.tflite
        eon_mem = getattr(low_end.memory, 'eon', None)
        ram_bytes = tflite_mem.ram
        rom_bytes = tflite_mem.rom
        arena_size_bytes = getattr(tflite_mem, 'arena_size', None)
        eon_ram_bytes = getattr(eon_mem, 'ram', None) if eon_mem else None
        eon_rom_bytes = getattr(eon_mem, 'rom', None) if eon_mem else None
        is_supported  = getattr(low_end, 'is_supported_on_mcu', None)
        ei_latency_ms = getattr(low_end, 'time_per_inference_ms', None)
        # Scale latency from EI 40 MHz reference to board 32 MHz clock
        # Scaling is linear for in-order M0+ cores: latency = cycles / clock_hz
        scale = HARDWARE_CONFIG['cortex_m0plus']['latency_scale']
        latency_ms = round(ei_latency_ms * scale, 3) if ei_latency_ms is not None else None

        return {
            'ram_bytes': ram_bytes,'rom_bytes': rom_bytes,'arena_size_bytes': arena_size_bytes,'eon_ram_bytes': eon_ram_bytes,'eon_rom_bytes': eon_rom_bytes,
            'latency_ms': latency_ms,'latency_ms_ei_raw': ei_latency_ms,'tflite_file_size_bytes': os.path.getsize(tflite_path),
            'is_supported': bool(is_supported) if is_supported is not None else None,
            'ram_kb': round(ram_bytes / 1024, 3),'rom_kb': round(rom_bytes / 1024, 3),
            'arena_size_kb': round(arena_size_bytes / 1024, 3) if arena_size_bytes else None,
            'eon_ram_kb': round(eon_ram_bytes / 1024, 3) if eon_ram_bytes else None,
            'eon_rom_kb': round(eon_rom_bytes / 1024, 3) if eon_rom_bytes else None,
            'tflite_file_size_kb': round(os.path.getsize(tflite_path) / 1024, 3),
            'ei_clock_mhz': HARDWARE_CONFIG['cortex_m0plus']['ei_clock_mhz'],
            'board_clock_mhz': HARDWARE_CONFIG['cortex_m0plus']['clock_mhz'],
        }
    except Exception as e:
        print(f"[EI] Parse low_end_mcu error: {e}")
        return None


def _empty_profile():
    return {'ram_bytes': None,'rom_bytes': None,'arena_size_bytes': None,'eon_ram_bytes': None,'eon_rom_bytes': None,'latency_ms': None,'tflite_file_size_bytes': None,
        'is_supported': None,'ram_kb': None,'rom_kb': None,'arena_size_kb': None,'eon_ram_kb': None,'eon_rom_kb': None,'tflite_file_size_kb': None,
    }


def _get_variant(resp):
    """
    Extracts int8 or float32 variant from a named-device profile response.
    """
    try:
        info = resp.model.profile_info
        return getattr(info, 'int8', None) or getattr(info, 'float32', None)
    except Exception:
        return None


def _get_low_end_mcu(resp):
    """
    Extracts the low-end MCU variant from a generic call.
    """
    try:
        info = resp.model.profile_info
        variant = getattr(info, 'int8', None) or getattr(info, 'float32', None)
        if variant is None:
            return None
        if not hasattr(variant, 'memory') or not hasattr(variant, 'time_per_inference_ms'):
            print(f"[EI] Variant missing expected fields. "
                  f"Attrs: {[a for a in dir(variant) if not a.startswith('_')]}")
            return None
        return variant
    except Exception as e:
        print(f"[EI] _get_low_end_mcu error: {e}")
        return None


def _profile_device(tflite_path, mcu_name, device_str, max_retries=3):
    """
    Profiles a single named device target with retry logic.
    Used for M4F and M7.
    """
    board = HARDWARE_CONFIG[mcu_name]['board']
    print(f"[EI] Profiling {mcu_name} ({device_str} / {board})...")
    for attempt in range(max_retries):
        try:
            resp = ei.model.profile(model=tflite_path, device=device_str)
            variant = _get_variant(resp)
            result = _parse_variant(variant, tflite_path)
            if result is not None:
                print(
                    f"RAM: {result['ram_kb']} KB | "f"ROM: {result['rom_kb']} KB | "f"Arena: {result['arena_size_kb']} KB | "f"Latency: {result['latency_ms']} ms | "
                    f"EON RAM: {result['eon_ram_kb']} KB | "f"EON ROM: {result['eon_rom_kb']} KB | "f"Supported: {result['is_supported']}"
                )
                return result
        except Exception as e:
            print(f"[EI] Attempt {attempt + 1}/{max_retries} failed for {mcu_name}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
    print(f"[EI] All attempts failed for {mcu_name} returning empty profile.")
    return _empty_profile()


def profile_tflite(tflite_path, max_retries=3):
    """
    Profiles a TFLite model against all three MCU targets.
    """
    results = {}

    # M4F and M7 
    for mcu_name, device_str in EI_DEVICE_MAP.items():
        results[mcu_name] = _profile_device(
            tflite_path, mcu_name, device_str, max_retries
        )

    # M0+ 
    board_m0 = HARDWARE_CONFIG['cortex_m0plus']['board']
    print(f"[EI] Profiling cortex_m0plus "
          f"(generic lowEndMcu / {board_m0}, scaled 40→32 MHz)...")

    profile_m0 = None
    for attempt in range(max_retries):
        try:
            resp = ei.model.profile(model=tflite_path)
            low_end = _get_low_end_mcu(resp)
            profile_m0 = _parse_low_end_mcu(low_end, tflite_path)
            if profile_m0 is not None:
                print(
                    f"RAM: {profile_m0['ram_kb']} KB | "f"ROM: {profile_m0['rom_kb']} KB | "f"Arena: {profile_m0['arena_size_kb']} KB | "
                    f"Latency (32 MHz): {profile_m0['latency_ms']} ms | "f"Latency (EI 40 MHz raw): {profile_m0['latency_ms_ei_raw']} ms | "
                    f"EON RAM: {profile_m0['eon_ram_kb']} KB | "f"EON ROM: {profile_m0['eon_rom_kb']} KB | "f"Supported: {profile_m0['is_supported']}"
                )
                break
        except Exception as e:
            print(f"[EI] Attempt {attempt + 1}/{max_retries} failed "
                  f"for cortex_m0plus: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    if profile_m0 is None:
        print(f"[EI] All attempts failed for cortex_m0plus")

    results['cortex_m0plus'] = profile_m0 if profile_m0 is not None else _empty_profile()
    return results


def assess_deployability(hardware_profile, hardware_config):
    """
    Checks whether a model fits within each board's RAM and flash budget.
    Budgets from hardware_config (board internal silicon specs):
    -  cortex_m0plus NUCLEO-L073RZ (STM32L073RZT6): 20 KB RAM / 192 KB flash
    -  cortex_m4f NUCLEO-F401RE (STM32F401RET6): 96 KB RAM / 512 KB flash
    -  cortex_m7 NUCLEO-F746ZG (STM32F746ZGT6): 320 KB RAM / 1024 KB flash
    """
    results = {}

    for mcu_name, budget in hardware_config.items():
        profile = hardware_profile.get(mcu_name, {})

        ram_kb = profile.get('ram_kb')
        rom_kb = profile.get('rom_kb')

        ram_ok = ram_kb is not None and ram_kb <= budget['ram_kb']
        rom_ok = rom_kb is not None and rom_kb <= budget['flash_kb']

        results[mcu_name] = {'deployable': ram_ok and rom_ok,'ram_ok': ram_ok,'rom_ok': rom_ok,'ram_used_kb': ram_kb,'rom_used_kb': rom_kb,
                             'ram_budget_kb': budget['ram_kb'],'rom_budget_kb': budget['flash_kb'],}

        print(
            f"{mcu_name}: RAM {ram_kb} KB / {budget['ram_kb']} KB | "f"ROM {rom_kb} KB / {budget['flash_kb']} KB | "f"Deployable: {ram_ok and rom_ok}"
        )

    return results


def compute_pareto_frontier(results_list):
    """
    Identifies non-dominated pipelines on the (accuracy, ROM) objective pair.
    A pipeline is dominated if another achieves equal or higher accuracy at equal or lower ROM.
    """
    candidates = [
        r for r in results_list
        if r.get('tflite_metrics') and r.get('hardware_profile')
    ]

    points = []
    for r in candidates:
        acc = r['tflite_metrics']['accuracy']
        rom_kb = r.get('tflite_size_kb')
        if rom_kb is None:
            continue
        points.append({'pipeline': r['pipeline'],'domain': r['domain'],'seed': r['seed'],'accuracy': acc,'rom_kb': rom_kb,})

    pareto = []
    for candidate in points:
        dominated = any(
            other['accuracy'] >= candidate['accuracy']
            and other['rom_kb'] <= candidate['rom_kb']
            and (other['accuracy'] > candidate['accuracy'] or other['rom_kb'] < candidate['rom_kb'])
            for other in points
            if other is not candidate
        )
        if not dominated:
            pareto.append(candidate)

    pareto.sort(key=lambda x: x['rom_kb'])
    return pareto