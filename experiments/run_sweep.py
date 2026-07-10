import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from experiments.config import (BASE_CONFIG,DOMAINS_CONFIG,HARDWARE_CONFIG,SWEEP_CONFIG,)
from experiments.pruning_sweep import run_pruning_sweep


def main():
    parser = argparse.ArgumentParser(description='Run pruning ratio sweep for a given domain.')
    parser.add_argument('--domain',type=str,required=True,choices=['timeseries', 'image'],help="Domain to sweep: 'timeseries' (TCN/UCI HAR) or 'image' (ResNet-8/CIFAR-10).")
    args = parser.parse_args()
    domain = args.domain
    config = {**BASE_CONFIG, **DOMAINS_CONFIG[domain]}

    sweep_config = {
        **SWEEP_CONFIG,
        'domain': domain,
        'teacher_path': DOMAINS_CONFIG[domain]['teacher_path'],
    }

    print(f"\n[Sweep] Domain : {domain}")
    print(f"[Sweep] Teacher path : {sweep_config['teacher_path']}")
    print(f"[Sweep] Ratios : {sweep_config['prune_ratios']}")
    print(f"[Sweep] Output dir : {sweep_config['output_dir']}")

    if not os.path.exists(sweep_config['teacher_path']):
        raise FileNotFoundError(
            f"\n[Error] Teacher checkpoint not found: {sweep_config['teacher_path']}\n"
            f"Run first: python experiments/train_teachers.py --domain {domain}"
        )

    results, summary = run_pruning_sweep(
        sweep_config=sweep_config,
        base_config=config,
        hardware_config=HARDWARE_CONFIG,
    )

    print(f"\n[Sweep complete]")
    print(f"Inspect sweep_results/sweep_summary_{domain}.csv (pruning ratio).")
    print(f"Then update PRUNE_RATIO in experiments/config.py before running pipelines.")


if __name__ == '__main__':
    main()