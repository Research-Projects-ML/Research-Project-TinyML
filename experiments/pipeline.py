import os
import gc
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

from compression.distillation  import apply_knowledge_distillation
from compression.pruning       import apply_structured_pruning
from compression.quantization  import apply_ptq, dequantize_to_float, requantize

from datasets.image_loader      import load_image_data
from datasets.timeseries_loader import load_timeseries_data

from models.image.resnet8  import get_teacher as get_image_teacher
from models.image.resnet8  import get_student as get_image_student
from models.timeseries.tcn import get_teacher as get_ts_teacher
from models.timeseries.tcn import get_student as get_ts_student

from evaluation.model_metrics    import (
    evaluate_keras_model,
    evaluate_tflite_model,
    compute_accuracy_drop,
    compute_l1_sensitivity,
    serialize_history,
)
from evaluation.hardware_metrics import (
    profile_tflite_with_ei,
    assess_deployability,
    compute_pareto_frontier,
)


def set_global_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_merged_config(base_config, domain_config):
    merged = {**base_config, **domain_config}
    merged['kd_patience'] = merged.get('early_stop_patience', 10)
    return merged


def load_data(domain, config, seed):
    if domain == 'image':
        return load_image_data(config, seed)
    elif domain == 'timeseries':
        return load_timeseries_data(config, seed)
    else:
        raise ValueError(f"Unsupported domain: {domain}")


def build_fresh_student(domain, config):
    if domain == 'image':
        return get_image_student(config)
    elif domain == 'timeseries':
        return get_ts_student(config)
    else:
        raise ValueError(f"Unsupported domain: {domain}")


def load_keras_model(path):
    from models.timeseries.tcn import CausalDilatedConv1D, LastTimestep
    custom_objects = {
        'CausalDilatedConv1D': CausalDilatedConv1D,
        'LastTimestep':        LastTimestep,
    }
    return keras.models.load_model(path, custom_objects=custom_objects)


def save_keras_model(model, path):
    model.save(path)
    print(f"  [Checkpoint] Saved → {path}")


def save_result(result, eval_dir, pipeline_name):
    os.makedirs(eval_dir, exist_ok=True)
    path = os.path.join(eval_dir, f'{pipeline_name}.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  [Result] Saved → {path}")


def result_exists(eval_dir, pipeline_name):
    return os.path.exists(os.path.join(eval_dir, f'{pipeline_name}.json'))


def finetune_model(model, train_dataset, val_dataset, config):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['finetune_lr']),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=config.get('early_stop_patience', 10),
            restore_best_weights=True
        )
    ]
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config['finetune_epochs'],
        callbacks=callbacks,
        verbose=1
    )
    return model, history.history


def train_baseline(model, train_dataset, val_dataset, config):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['finetune_lr']),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=config.get('early_stop_patience', 10),
            restore_best_weights=True
        )
    ]
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.get('kd_epochs', 50),
        callbacks=callbacks,
        verbose=1
    )
    return model, history.history


def _apply_stage(
    stage,
    current_model,
    float_checkpoint_before_q,
    teacher,
    train_dataset,
    val_dataset,
    config,
    domain,
    pipeline_dir,
    pipeline_name,
    stage_idx,
):
    stage_label  = f"{pipeline_name}_stage{stage_idx}_{stage}"
    stage_history = None

    if stage == 'P':
        print(f"  [P] Structured pruning → fine-tuning...")
        pruned = apply_structured_pruning(
            current_model, config['prune_ratio'], domain
        )
        pruned, ft_history = finetune_model(
            pruned, train_dataset, val_dataset, config
        )
        stage_history = ft_history
        ckpt_path = os.path.join(pipeline_dir, f'{stage_label}.keras')
        save_keras_model(pruned, ckpt_path)
        return pruned, float_checkpoint_before_q, None, stage_history

    elif stage == 'KD':
        print(f"  [KD] Knowledge distillation...")
        student_kd, kd_history = apply_knowledge_distillation(
            teacher=teacher,
            student=current_model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_classes=config['num_classes']
        )
        stage_history = kd_history
        ckpt_path = os.path.join(pipeline_dir, f'{stage_label}.keras')
        save_keras_model(student_kd, ckpt_path)
        return student_kd, float_checkpoint_before_q, None, stage_history

    elif stage == 'Q':
        print(f"  [Q] Post-training quantization...")
        float_checkpoint_before_q = current_model
        calibration_dataset       = train_dataset.take(10)
        tflite_path               = os.path.join(
            pipeline_dir, f'{stage_label}.tflite'
        )
        tflite_bytes, _ = apply_ptq(
            current_model, calibration_dataset, tflite_path
        )
        return current_model, float_checkpoint_before_q, (tflite_bytes, tflite_path), stage_history

    else:
        raise ValueError(f"Unknown stage: {stage}")


def run_pipeline(
    pipeline,
    domain,
    config,
    hardware_config,
    teacher,
    baseline_metrics,
    train_dataset,
    val_dataset,
    test_dataset,
    pipeline_dir,
    eval_dir,
    seed,
):
    pipeline_name = pipeline['name']
    stages        = pipeline['stages']

    print(f"\n{'='*60}")
    print(f"Pipeline : {pipeline_name} | Domain : {domain} | Seed : {seed}")
    print(f"Stages   : {stages if stages else 'baseline'}")
    print(f"{'='*60}")

    if result_exists(eval_dir, pipeline_name):
        print(f"  [Skip] Already complete — loading existing result.")
        path = os.path.join(eval_dir, f'{pipeline_name}.json')
        with open(path) as f:
            return json.load(f)

    set_global_seed(seed)

    current_model             = build_fresh_student(domain, config)
    float_checkpoint_before_q = None
    tflite_artifact           = None
    all_stage_histories       = {}

    if not stages:
        print("  [Baseline] Training student from scratch...")
        current_model, history = train_baseline(
            current_model, train_dataset, val_dataset, config
        )
        all_stage_histories['baseline_train'] = serialize_history(history)
        ckpt_path = os.path.join(pipeline_dir, f'{pipeline_name}_student.keras')
        save_keras_model(current_model, ckpt_path)

    for idx, stage in enumerate(stages):
        needs_float_revert = (
            stage in ('P', 'KD')
            and tflite_artifact is not None
        )

        if needs_float_revert:
            print(
                f"  [Revert] Q was applied earlier — "
                f"reverting to float checkpoint for stage {stage}..."
            )
            current_model   = dequantize_to_float(
                tflite_artifact[1],
                float_checkpoint_before_q
            )
            tflite_artifact = None

        current_model, float_checkpoint_before_q, tflite_artifact, stage_history = _apply_stage(
            stage=stage,
            current_model=current_model,
            float_checkpoint_before_q=float_checkpoint_before_q,
            teacher=teacher,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            domain=domain,
            pipeline_dir=pipeline_dir,
            pipeline_name=pipeline_name,
            stage_idx=idx,
        )

        if stage_history is not None:
            all_stage_histories[f'stage{idx}_{stage}'] = serialize_history(
                stage_history
            )

    if tflite_artifact is None:
        print("  [Final Q] Applying PTQ to final float model...")
        calibration_dataset = train_dataset.take(10)
        tflite_path         = os.path.join(
            pipeline_dir, f'{pipeline_name}_final.tflite'
        )
        tflite_bytes, _  = apply_ptq(
            current_model, calibration_dataset, tflite_path
        )
        tflite_artifact  = (tflite_bytes, tflite_path)

    tflite_bytes, tflite_path = tflite_artifact

    print("  [Eval] Evaluating float model...")
    float_metrics = evaluate_keras_model(current_model, test_dataset)

    print("  [Eval] Evaluating TFLite model...")
    tflite_metrics = evaluate_tflite_model(tflite_bytes, test_dataset)

    print("  [Eval] Computing accuracy drop vs baseline...")
    float_drop  = compute_accuracy_drop(baseline_metrics, float_metrics)
    tflite_drop = compute_accuracy_drop(baseline_metrics, tflite_metrics)

    print("  [Eval] Computing L1 sensitivity profile...")
    l1_sensitivity = compute_l1_sensitivity(current_model, domain)

    print("  [EI] Profiling on hardware targets...")
    hardware_profile = profile_tflite_with_ei(tflite_path, hardware_config)

    print("  [Eval] Assessing deployability...")
    deployability = assess_deployability(hardware_profile, hardware_config)

    result = {
        'pipeline':          pipeline_name,
        'domain':            domain,
        'seed':              seed,
        'stages':            stages,
        'float_metrics':     float_metrics,
        'tflite_metrics':    tflite_metrics,
        'float_drop':        float_drop,
        'tflite_drop':       tflite_drop,
        'l1_sensitivity':    l1_sensitivity,
        'hardware_profile':  hardware_profile,
        'deployability':     deployability,
        'tflite_size_kb':    round(os.path.getsize(tflite_path) / 1024, 3),
        'stage_histories':   all_stage_histories,
    }

    print(f"\n  [Summary]")
    print(f"    Float  acc : {float_metrics['accuracy']:.4f} | F1: {float_metrics['macro_f1']:.4f}")
    print(f"    TFLite acc : {tflite_metrics['accuracy']:.4f} | F1: {tflite_metrics['macro_f1']:.4f}")
    print(f"    Acc drop   : {tflite_drop['accuracy_drop']:.4f}")
    print(f"    Size (KB)  : {result['tflite_size_kb']}")
    for mcu, d in deployability.items():
        print(f"    {mcu}: RAM {d['ram_used_kb']}KB / {d['ram_budget_kb']}KB | "
              f"ROM {d['rom_used_kb']}KB / {d['rom_budget_kb']}KB | "
              f"Deployable: {d['deployable']}")

    save_result(result, eval_dir, pipeline_name)

    del current_model
    gc.collect()

    return result


def run_all_pipelines(
    domain,
    pipelines,
    base_config,
    domains_config,
    hardware_config,
    teacher_path,
    pipeline_dir,
    eval_dir,
    seed,
):
    config = get_merged_config(base_config, domains_config[domain])

    print(f"\n[Setup] Loading data — domain={domain}, seed={seed}...")
    train_dataset, val_dataset, test_dataset = load_data(domain, config, seed)

    print(f"[Setup] Loading teacher from {teacher_path}...")
    teacher           = load_keras_model(teacher_path)
    teacher.trainable = False

    print(f"[Setup] Computing baseline metrics from teacher...")
    baseline_metrics = evaluate_keras_model(teacher, test_dataset)
    print(f"  Teacher accuracy : {baseline_metrics['accuracy']:.4f}")
    print(f"  Teacher macro F1 : {baseline_metrics['macro_f1']:.4f}")

    all_results = []

    for pipeline in pipelines:
        result = run_pipeline(
            pipeline=pipeline,
            domain=domain,
            config=config,
            hardware_config=hardware_config,
            teacher=teacher,
            baseline_metrics=baseline_metrics,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            pipeline_dir=pipeline_dir,
            eval_dir=eval_dir,
            seed=seed,
        )
        if result:
            all_results.append(result)

    print(f"\n[Analysis] Computing Pareto frontier across all pipelines...")
    pareto = compute_pareto_frontier(all_results)
    pareto_path = os.path.join(eval_dir, 'pareto_frontier.json')
    with open(pareto_path, 'w') as f:
        json.dump(pareto, f, indent=2)
    print(f"  Pareto frontier saved → {pareto_path}")
    print(f"  Pareto-optimal pipelines: {[p['pipeline'] for p in pareto]}")

    del teacher
    gc.collect()

    print(f"\n[Done] All pipelines complete — domain={domain}, seed={seed}.")
    return all_results