import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np

from experiments.config import (BASE_CONFIG,DOMAINS_CONFIG,HARDWARE_CONFIG,PIPELINES,ORDERING_SEEDS,PIPELINE_DIR,EVAL_DIR,)
from experiments.pipeline import run_all_pipelines


def aggregate_results(all_seed_results, pipelines):
    """
    Aggregates per-seed results into mean and std across seeds.
    For each pipeline, collects tflite accuracy and macro F1 across seeds and computes descriptive statistics.
    """
    # Group results by pipeline name
    by_pipeline = {p['name']: [] for p in pipelines}

    for seed_results in all_seed_results:
        for result in seed_results:
            name = result['pipeline']
            if name in by_pipeline:
                by_pipeline[name].append(result)

    aggregated = []

    for pipeline_name, results in by_pipeline.items():
        if not results:
            continue

        accs = [r['tflite_metrics']['accuracy']  for r in results]
        f1s = [r['tflite_metrics']['macro_f1']  for r in results]
        drops = [r['tflite_drop']['accuracy_drop'] for r in results]

        hw_profile = results[0].get('hardware_profile', {})
        deployability = results[0].get('deployability', {})
        tflite_size = results[0].get('tflite_size_kb')
        stages = results[0].get('stages', [])

        aggregated.append({'pipeline': pipeline_name,'stages': stages,'seeds_run': len(results),'mean_accuracy': round(float(np.mean(accs)), 4),
            'std_accuracy': round(float(np.std(accs)), 4),'mean_f1': round(float(np.mean(f1s)), 4),'std_f1': round(float(np.std(f1s)), 4),
            'mean_acc_drop': round(float(np.mean(drops)), 4),'std_acc_drop': round(float(np.std(drops)), 4),'tflite_size_kb': tflite_size,
            'hardware_profile': hw_profile,'deployability': deployability,
            'per_seed': [
                {'seed': r['seed'],'accuracy': r['tflite_metrics']['accuracy'],'macro_f1': r['tflite_metrics']['macro_f1'],'acc_drop': r['tflite_drop']['accuracy_drop'],}
                for r in results
            ],
        })

    aggregated.sort(key=lambda x: x['mean_accuracy'], reverse=True)
    return aggregated


def print_summary(aggregated, domain):
    """
    Prints a ranked summary table of all pipelines after aggregation.
    Columns: rank, pipeline, mean acc ± std, mean F1 ± std, size KB, seeds.
    """
    header = (
        f"{'Rank':>4} | {'Pipeline':<12} | {'Mean Acc':>8} | {'Std':>6} | "
        f"{'Mean F1':>7} | {'Std':>6} | {'Size KB':>7} | {'Seeds':>5}"
    )
    print(f"\n{'='*len(header)}")
    print(f"Ordering Results Summary — domain={domain}")
    print(f"{'='*len(header)}")
    print(header)
    print('-' * len(header))

    for rank, row in enumerate(aggregated, 1):
        print(
            f"{rank:>4} | "
            f"{row['pipeline']:<12} | "
            f"{row['mean_accuracy']:>8.4f} | "
            f"{row['std_accuracy']:>6.4f} | "
            f"{row['mean_f1']:>7.4f} | "
            f"{row['std_f1']:>6.4f} | "
            f"{str(row['tflite_size_kb']):>7} | "
            f"{row['seeds_run']:>5}"
        )


def main():
    parser = argparse.ArgumentParser(description='Run all compression ordering pipelines for a given domain.')
    parser.add_argument('--domain',type=str,required=True,choices=['timeseries', 'image'],help="Domain to run: 'timeseries' or 'image'")
    parser.add_argument('--seeds',type=int,nargs='+',default=None,help="Example: --seeds 0 1 to run only the first two seeds.")
    parser.add_argument(
        '--pipelines',
        type=str,
        nargs='+',
        default=None,
        help=(
            "Pipeline names to run. Defaults to all pipelines in config.py. "
            "Example: --pipelines P_KD_QAT KD_P_QAT to run specific orderings only."
        )
    )
    args = parser.parse_args()
    domain = args.domain
    seeds = args.seeds if args.seeds is not None else ORDERING_SEEDS

    # Filter pipelines if specific names were requested
    if args.pipelines is not None:
        pipelines = [p for p in PIPELINES if p['name'] in args.pipelines]
        missing = set(args.pipelines) - {p['name'] for p in pipelines}
        if missing:
            raise ValueError(
                f"Unknown pipeline names: {missing}. "
                f"Valid names: {[p['name'] for p in PIPELINES]}"
            )
    else:
        pipelines = PIPELINES

    config = {**BASE_CONFIG, **DOMAINS_CONFIG[domain]}
    teacher_path = DOMAINS_CONFIG[domain]['teacher_path']
    pipeline_dir  = os.path.join(PIPELINE_DIR, domain)
    eval_dir = os.path.join(EVAL_DIR,domain)

    os.makedirs(pipeline_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    print(f"\n[Pipelines] Domain: {domain}")
    print(f"[Pipelines] Seeds: {seeds}")
    print(f"[Pipelines] Pipelines: {[p['name'] for p in pipelines]}")
    print(f"[Pipelines] Teacher: {teacher_path}")
    print(f"[Pipelines] Pipeline dir: {pipeline_dir}")
    print(f"[Pipelines] Eval dir: {eval_dir}")

    # Verify teacher exists before starting any seed
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(
            f"\n[Error] Teacher checkpoint not found: {teacher_path}\n"
            f"Run first: python experiments/train_teachers.py --domain {domain}"
        )

    all_seed_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed} / {seeds}")
        print(f"{'='*60}")

        # Per-seed subdirectories keep checkpoints from different seeds separate
        seed_pipeline_dir = os.path.join(pipeline_dir, f'seed_{seed}')
        seed_eval_dir = os.path.join(eval_dir, f'seed_{seed}')

        os.makedirs(seed_pipeline_dir, exist_ok=True)
        os.makedirs(seed_eval_dir, exist_ok=True)

        seed_results = run_all_pipelines(
            domain=domain,pipelines=pipelines,base_config=BASE_CONFIG,domains_config=DOMAINS_CONFIG,hardware_config=HARDWARE_CONFIG,
            teacher_path=teacher_path,pipeline_dir=seed_pipeline_dir,eval_dir=seed_eval_dir,seed=seed,
        )

        all_seed_results.append(seed_results)

    # Aggregate across seeds and save
    print(f"\n[Aggregation] Computing mean ± std across {len(seeds)} seeds...")
    aggregated = aggregate_results(all_seed_results, pipelines)
    agg_path = os.path.join(eval_dir, 'aggregated_results.json')

    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"Aggregated results saved to {agg_path}")
    
    print_summary(aggregated, domain)
    print(f"\n[Done] All pipelines complete: domain={domain}, seeds={seeds}.")


if __name__ == '__main__':
    main()