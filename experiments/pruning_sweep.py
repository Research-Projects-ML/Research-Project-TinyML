import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import json
import csv
import numpy as np
import tensorflow as tf
import keras

from experiments.utils import (set_seed, load_data, get_calibration_batches,result_exists, save_result, get_steps_per_epoch,)
from experiments.config import (BASE_CONFIG, DOMAINS_CONFIG, HARDWARE_CONFIG,SWEEP_CONFIG as _BASE_SWEEP_CONFIG,)
from experiments.pipeline import get_merged_config, _train_model
from compression.pruning import apply_structured_pruning
from compression.quantization import apply_ptq
from evaluation.model_metrics import evaluate_keras_model, evaluate_tflite_model
from evaluation.hardware_metrics import profile_tflite, assess_deployability

SWEEP_CONFIG = {
    **_BASE_SWEEP_CONFIG,
    'domain': 'image',
}

def _result_filename(domain, ratio):
    ratio_str = f"{ratio:.2f}".replace('.', 'p')
    return f"sweep_{domain}_ratio_{ratio_str}.json"


def _result_exists_sweep(output_dir, domain, ratio):
    return os.path.exists(os.path.join(output_dir, _result_filename(domain, ratio)))


def _save_result_sweep(result, output_dir, domain, ratio):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, _result_filename(domain, ratio))
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  [Saved] {path}")


def _build_result(prune_ratio, domain, seed, float_metrics, tflite_metrics,baseline_metrics, hardware_profile, deployability, tflite_path):
    return {
        'prune_ratio': prune_ratio,
        'domain': domain,
        'seed': seed,
        'float_metrics': float_metrics,
        'tflite_metrics': tflite_metrics,
        'acc_drop_float': round(baseline_metrics['accuracy'] - float_metrics['accuracy'],  4),
        'acc_drop_tflite': round(baseline_metrics['accuracy'] - tflite_metrics['accuracy'], 4),
        'tflite_size_kb': round(os.path.getsize(tflite_path) / 1024, 3),
        'hardware_profile': hardware_profile,
        'deployability': deployability,
    }


def train_student(domain, config, train_dataset, val_dataset, seed):
    """
    Trains a fresh student from random initialisation (GlorotUniform).
    """
    set_seed(seed)
    if domain == 'timeseries':
        from models.timeseries.tcn import get_student as get_ts_student
        student = get_ts_student(config)
    elif domain == 'image':
        from models.image.resnet8 import get_student as get_image_student
        student = get_image_student(config)
    else:
        raise ValueError(f"Unknown domain: '{domain}'.")
    student, _ = _train_model(
        student, train_dataset, val_dataset, config,
        lr_key='train_lr', epochs_key='train_epochs',
    )
    return student


def finetune(model, train_dataset, val_dataset, config):
    """Fine-tunes a pruned model to recover accuracy lost from channel removal."""
    model, _ = _train_model(
        model, train_dataset, val_dataset, config,
        lr_key='finetune_lr', epochs_key='finetune_epochs',
    )
    return model


def run_ratio(prune_ratio, domain, config, student, baseline_metrics,train_dataset, val_dataset, test_dataset, output_dir, seed,):
    """
    Runs one prune ratio: prune student → fine-tune → PTQ → evaluate → profile.
    Accuracy drop is student-vs-student. Skips if result JSON already exists.
    """
    print(f"\n{'='*55}")
    print(f"  Ratio: {prune_ratio:.2f} | Domain: {domain} | Seed: {seed}")
    print(f"{'='*55}")

    if _result_exists_sweep(output_dir, domain, prune_ratio):
        print(f"  [Skip] Result already exists.")
        path = os.path.join(output_dir, _result_filename(domain, prune_ratio))
        with open(path) as f:
            return json.load(f)

    set_seed(seed)

    tflite_dir = os.path.join(output_dir, 'tflite')
    os.makedirs(tflite_dir, exist_ok=True)

    if prune_ratio == 0.0:
        print("  [Baseline] No pruning — evaluating uncompressed student.")
        float_metrics = baseline_metrics
        tflite_path = os.path.join(tflite_dir, f"baseline_{domain}.tflite")
        calib_batches = get_calibration_batches(train_dataset, config)
        tflite_bytes, _ = apply_ptq(student, calib_batches, tflite_path)
        tflite_metrics = evaluate_tflite_model(tflite_bytes, test_dataset)
        hardware_profile = profile_tflite(tflite_path)
        deployability = assess_deployability(hardware_profile, HARDWARE_CONFIG)

        result = _build_result(
            prune_ratio, domain, seed,
            float_metrics, tflite_metrics,
            baseline_metrics, hardware_profile,
            deployability, tflite_path,
        )
        _save_result_sweep(result, output_dir, domain, prune_ratio)
        return result

    print(f"[Prune] Applying structured pruning at ratio {prune_ratio:.2f}...")
    pruned_model = apply_structured_pruning(student, prune_ratio, domain)
    print(f"[Finetune] Fine-tuning pruned model...")
    pruned_model = finetune(pruned_model, train_dataset, val_dataset, config)
    float_metrics = evaluate_keras_model(pruned_model, test_dataset)
    print(
        f"Float Acc: {float_metrics['accuracy']:.4f} | "
        f"F1: {float_metrics['macro_f1']:.4f}"
    )

    ratio_str = f"{prune_ratio:.2f}".replace('.', 'p')
    tflite_path = os.path.join(tflite_dir, f"pruned_{domain}_ratio_{ratio_str}.tflite")

    calib_batches = get_calibration_batches(train_dataset, config)
    tflite_bytes, _ = apply_ptq(pruned_model, calib_batches, tflite_path)
    tflite_metrics = evaluate_tflite_model(tflite_bytes, test_dataset)
    hardware_profile = profile_tflite(tflite_path)
    deployability = assess_deployability(hardware_profile, HARDWARE_CONFIG)

    print(f"TFLite Acc: {tflite_metrics['accuracy']:.4f} | "f"F1: {tflite_metrics['macro_f1']:.4f}")

    result = _build_result(
        prune_ratio, domain, seed,
        float_metrics, tflite_metrics,
        baseline_metrics, hardware_profile,
        deployability, tflite_path,
    )
    _save_result_sweep(result, output_dir, domain, prune_ratio)

    del pruned_model
    keras.backend.clear_session()
    gc.collect()

    return result


def build_summary(results, output_dir, domain):
    """Prints a ranked table and writes a CSV. Both M0+ and M4F columns included."""
    summary_rows = []

    for r in results:
        hw = r.get('hardware_profile', {})
        m0 = hw.get('cortex_m0plus', {})
        m4f = hw.get('cortex_m4f', {})
        dep_m0 = r.get('deployability', {}).get('cortex_m0plus', {})
        dep_m4f = r.get('deployability', {}).get('cortex_m4f', {})

        summary_rows.append({
            'prune_ratio': r['prune_ratio'],
            'float_accuracy': r['float_metrics']['accuracy'],
            'tflite_accuracy': r['tflite_metrics']['accuracy'],
            'float_f1': r['float_metrics']['macro_f1'],
            'tflite_f1': r['tflite_metrics']['macro_f1'],
            'acc_drop_tflite': r['acc_drop_tflite'],
            'tflite_size_kb': r['tflite_size_kb'],
            'ram_kb_m0': m0.get('ram_kb'),
            'rom_kb_m0': m0.get('rom_kb'),
            'latency_ms_m0': m0.get('latency_ms'),
            'deployable_m0': dep_m0.get('deployable'),
            'ram_kb_m4f': m4f.get('ram_kb'),
            'rom_kb_m4f': m4f.get('rom_kb'),
            'latency_ms_m4f': m4f.get('latency_ms'),
            'deployable_m4f': dep_m4f.get('deployable'),
        })

    header = (
        f"{'Ratio':>6}  | {'Float':>7} | {'TFLite':>7} | "
        f"{'Drop':>6}   | {'KB':>6} | "
        f"{'M0+RAM':>7} | {'M0+Lat':>7} | {'M0+OK':>6} | "
        f"{'M4FRAM':>7} | {'M4FLat':>7} | {'M4FOK':>6}"
    )
    print(f"\n{'='*len(header)}")
    print(f"Pruning Sweep Summary — domain={domain}")
    print(f"{'='*len(header)}")
    print(header)
    print('-' * len(header))

    for row in summary_rows:
        print(
            f"{row['prune_ratio']:>6.2f} | "
            f"{row['float_accuracy']:>7.4f} | "
            f"{row['tflite_accuracy']:>7.4f} | "
            f"{row['acc_drop_tflite']:>6.4f} | "
            f"{row['tflite_size_kb']:>6.2f} | "
            f"{str(row['ram_kb_m0']):>7} | "
            f"{str(row['latency_ms_m0']):>7} | "
            f"{str(row['deployable_m0']):>6} | "
            f"{str(row['ram_kb_m4f']):>7} | "
            f"{str(row['latency_ms_m4f']):>7} | "
            f"{str(row['deployable_m4f']):>6}"
        )

    csv_path = os.path.join(output_dir, f"sweep_summary_{domain}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n  [CSV] Saved → {csv_path}")

    return summary_rows


def find_knee_ratio(summary_rows):
    """
    Identifies the pruning ratio at the knee of the accuracy-vs-ratio curve
    using perpendicular distance from the line connecting first and last points.
    Always inspect the full table before using this recommendation.
    """
    candidates = [
        r for r in summary_rows
        if r['prune_ratio'] > 0 and r['tflite_accuracy'] is not None
    ]
    if len(candidates) < 2:
        return None

    ratios = np.array([r['prune_ratio']     for r in candidates])
    accs   = np.array([r['tflite_accuracy'] for r in candidates])

    p1 = np.array([ratios[0],  accs[0]])
    p2 = np.array([ratios[-1], accs[-1]])
    line_vec = p2 - p1

    distances = []
    for ratio, acc in zip(ratios, accs):
        point = np.array([ratio, acc])
        t = np.dot(point - p1, line_vec) / (np.dot(line_vec, line_vec) + 1e-8)
        projection = p1 + t * line_vec
        distances.append(np.linalg.norm(point - projection))

    knee_idx = int(np.argmax(distances))
    return candidates[knee_idx]['prune_ratio']


def plot_sweep(domain, sweep_results_dir, output_path=None):
    """
    Plots TFLite accuracy and M0+ RAM vs prune ratio.
    Filled circle = M0+ deployable. Open circle = exceeds M0+ budget.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("  [Plot] matplotlib not available — skipping.")
        return None

    ratios, accs, rams, deployable = [], [], [], []

    for ratio_str, ratio_val in [('0p00', 0.0), ('0p10', 0.1), ('0p20', 0.2), ('0p30', 0.3),('0p40', 0.4), ('0p50', 0.5), ('0p60', 0.6),]:
        path = os.path.join(sweep_results_dir, f'sweep_{domain}_ratio_{ratio_str}.json')
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        ratios.append(ratio_val)
        accs.append(d['tflite_metrics']['accuracy'] * 100)
        hw = d.get('hardware_profile', {}).get('cortex_m0plus', {})
        rams.append(hw.get('ram_kb', 0))
        dep = d.get('deployability', {}).get('cortex_m0plus', {})
        deployable.append(dep.get('deployable', False))

    if not ratios:
        print("  [Plot] No result files found — run the sweep first.")
        return None

    ratios = np.array(ratios)
    accs = np.array(accs)
    rams = np.array(rams)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    ax1.plot(ratios, accs, color='#3a6cbf', linewidth=2.0, zorder=3,label='TFLite Accuracy')
    for r, a, dep in zip(ratios, accs, deployable):
        if dep:
            ax1.plot(r, a, 'o', color='#3a6cbf', markersize=10, zorder=4)
        else:
            ax1.plot(r, a, 'o', color='white', markersize=9, zorder=4,markeredgecolor='#3a6cbf', markeredgewidth=1.8)
    ax2.plot(ratios, rams, '--', color='#c8742a', linewidth=1.8,zorder=2, label='M0+ RAM (KB)')
    ax2.plot(ratios, rams, 'o', color='#c8742a', markersize=6, zorder=3)
    ax2.axhline(y=32, color='#c8742a', linewidth=0.8,linestyle=':', alpha=0.7, label='M0+ budget (32 KB)')
    ax1.set_xlabel('Pruning Ratio', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11, color='#3a6cbf')
    ax2.set_ylabel('RAM (KB)',     fontsize=11, color='#c8742a')
    ax1.tick_params(axis='y', labelcolor='#3a6cbf')
    ax2.tick_params(axis='y', labelcolor='#c8742a')
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax1.set_xticks(ratios)
    ax1.set_ylim(np.floor(min(accs)) - 1, np.ceil(max(accs)) + 1)
    ax2.set_ylim(np.floor(min(rams) / 5) * 5 - 5,np.ceil(max(rams)  / 5) * 5 + 5,)
    ax1.set_facecolor('#eef1f8')
    fig.patch.set_facecolor('white')
    ax1.grid(True, color='white', linewidth=0.8, zorder=0)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,loc='lower left', fontsize=9, framealpha=0.9)
    plt.title(f'Pruning Ratio Sweep — {domain.capitalize()}', fontsize=12, pad=10)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  [Saved plot] {output_path}")
    else:
        plt.show()

    return fig


def run_pruning_sweep(sweep_config=None):
    cfg = sweep_config or SWEEP_CONFIG
    domain = cfg['domain']
    seed = cfg['seed']
    output_dir = cfg['output_dir']
    ratios = cfg['prune_ratios']

    config = get_merged_config(BASE_CONFIG, DOMAINS_CONFIG[domain])

    print(f"\n[Pruning Sweep] Domain={domain} | Seed={seed}")
    print(f"Ratios : {ratios}")
    print(f"Output : {output_dir}")

    set_seed(seed)

    print(f"\n[Setup] Loading data...")
    train_dataset, val_dataset, test_dataset = load_data(domain, config, seed)

    print(f"\n[Setup] Training student from scratch...")
    student = train_student(domain, config, train_dataset, val_dataset, seed)

    print(f"\n[Setup] Computing student baseline metrics...")
    baseline_metrics = evaluate_keras_model(student, test_dataset)
    print(f"Student accuracy : {baseline_metrics['accuracy']:.4f}")
    print(f"Student macro F1 : {baseline_metrics['macro_f1']:.4f}")

    results = []
    for ratio in ratios:
        result = run_ratio(
            prune_ratio = ratio,
            domain = domain,
            config = config,
            student = student,
            baseline_metrics = baseline_metrics,
            train_dataset = train_dataset,
            val_dataset = val_dataset,
            test_dataset = test_dataset,
            output_dir = output_dir,
            seed = seed,
        )
        results.append(result)

    summary_rows = build_summary(results, output_dir, domain)
    knee = find_knee_ratio(summary_rows)
    if knee is not None:
        print(f"\n  [Knee] Recommended pruning ratio : {knee:.2f}")
        print(
            f"  Inspect the table above and choose based on MCU deployment constraints.\n"
            f"  Then update PRUNE_RATIO in experiments/config.py before running pipelines."
        )

    plot_path = os.path.join(output_dir, f"sweep_plot_{domain}.png")
    plot_sweep(domain, output_dir, output_path=plot_path)

    del student
    keras.backend.clear_session()
    gc.collect()

    print(f"\n[Done] Sweep complete. Results in: {output_dir}/")
    return results, summary_rows


if __name__ == '__main__':
    run_pruning_sweep()