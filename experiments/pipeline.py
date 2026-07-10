import os
import gc
import json
import keras
import numpy as np
import tensorflow as tf
from experiments.utils import (set_seed, load_data, load_keras_model, save_keras_model, result_exists, save_result, get_calibration_batches,)
from experiments.config import BASE_CONFIG, DOMAINS_CONFIG
from compression.distillation import apply_knowledge_distillation
from compression.pruning import apply_structured_pruning
from compression.quantization import apply_ptq, apply_qat
from models.image.resnet8 import get_student as get_image_student
from models.timeseries.tcn import get_student as get_ts_student
from evaluation.model_metrics import (evaluate_keras_model, evaluate_tflite_model,compute_accuracy_drop, compute_l1_sensitivity, serialize_history,)
from evaluation.hardware_metrics import (profile_tflite, assess_deployability, compute_pareto_frontier,)


def get_merged_config(base_config, domain_config):
    """
    Merges base config with domain-specific overrides.
    Domain config keys take precedence over base config keys.
    """
    return {**base_config, **domain_config}


def build_fresh_student(domain, config):
    """
    Called at start of every pipeline run, Randomly initialised student model for the given domain.
    Every pipeline gets afresh GlorotUniform initialisation so no ordering pipeline benefits from another pipeline's trained weights.
    """
    if domain == 'image':
        return get_image_student(config)
    elif domain == 'timeseries':
        return get_ts_student(config)
    else:
        raise ValueError(f"Unsupported domain: {domain}")


def stage_checkpoint_exists(pipeline_dir, stage_label):
    return os.path.exists(os.path.join(pipeline_dir, f'{stage_label}.keras'))


def _train_model(model, train_dataset, val_dataset, config, lr_key, epochs_key):
    """
    Shared training helper used by finetune_model and train_baseline.
    Avoids duplicating compile/fit/callback logic.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config[lr_key]),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        jit_compile=False,
    )
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=config.get('early_stop_patience', 10),restore_best_weights=True,)
    ]
    history = model.fit(train_dataset,validation_data=val_dataset,epochs=config[epochs_key],callbacks=callbacks,verbose=1,)
    return model, history.history


def finetune_model(model, train_dataset, val_dataset, config):
    return _train_model(
        model, train_dataset, val_dataset, config, lr_key='finetune_lr', epochs_key='finetune_epochs'
    )


def train_baseline(model, train_dataset, val_dataset, config):
    return _train_model(
        model, train_dataset, val_dataset, config, lr_key='train_lr', epochs_key='train_epochs'
    )


def _apply_stage( stage, current_model, teacher, train_dataset, val_dataset, config, domain, pipeline_dir, pipeline_name, stage_idx, stages,):
    stage_label = f"{pipeline_name}_stage{stage_idx}_{stage}"
    ckpt_path   = os.path.join(pipeline_dir, f'{stage_label}.keras')

    if stage_checkpoint_exists(pipeline_dir, stage_label):
        print(f"  [Resume] Stage {stage} checkpoint found — loading {ckpt_path}")
        resumed_model = load_keras_model(ckpt_path)
        print(f"  [Resume] Stage {stage} skipped — already complete.")
        return resumed_model, None

    if stage == 'P':
        print(f"[P] Structured pruning to fine-tuning...")
        pruned = apply_structured_pruning(
            current_model, config['prune_ratio'], domain
        )
        pruned, ft_history = finetune_model(
            pruned, train_dataset, val_dataset, config
        )
        save_keras_model(pruned, ckpt_path)
        return pruned, ft_history

    elif stage == 'KD':
        print(f"[KD] Knowledge distillation...")
        kd_config = dict(config)
        kd_config.pop('kd_lr_post_pruning', None)

        student_kd, kd_history = apply_knowledge_distillation(teacher=teacher,student=current_model,config=kd_config,train_dataset=train_dataset,
                                                              val_dataset=val_dataset,num_classes=config['num_classes'],)
        save_keras_model(student_kd, ckpt_path)
        return student_kd, kd_history

    elif stage == 'QAT':
        print(f"[QAT] Quantization-aware training...")
        qat_model, qat_history = apply_qat(
            current_model, train_dataset, val_dataset, config
        )
        save_keras_model(qat_model, ckpt_path)
        return qat_model, qat_history

    else:
        raise ValueError(
            f"Unknown stage: '{stage}'. "
            f"Valid stages are 'P', 'KD', 'QAT'. "
            f"PTQ is applied unconditionally after all stages."
        )


def run_pipeline(pipeline,domain,config,hardware_config,teacher,baseline_metrics,train_dataset,val_dataset,test_dataset,pipeline_dir,eval_dir,seed,):

    pipeline_name = pipeline['name']
    stages = pipeline['stages']

    print(f"\n{'='*60}")
    print(f"Pipeline : {pipeline_name} | Domain : {domain} | Seed : {seed}")
    print(f"Stages : {stages if stages else 'baseline (no compression)'}")
    print(f"{'='*60}")

    set_seed(seed)

    if result_exists(eval_dir, pipeline_name):
        print(f"[Skip] Already complete loading existing result.")
        path = os.path.join(eval_dir, f'{pipeline_name}.json')
        with open(path) as f:
            return json.load(f)

    current_model = build_fresh_student(domain, config)
    all_stage_histories = {}

    if not stages:
        print("[Baseline] Training student from scratch...")
        baseline_label = f"{pipeline_name}_student"
        baseline_ckpt = os.path.join(pipeline_dir, f'{baseline_label}.keras')

        if os.path.exists(baseline_ckpt):
            print(f"[Resume] Baseline checkpoint found — loading {baseline_ckpt}")
            current_model = load_keras_model(baseline_ckpt)
        else:
            current_model, history = train_baseline(current_model, train_dataset, val_dataset, config)
            all_stage_histories['baseline_train'] = serialize_history(history)
            save_keras_model(current_model, baseline_ckpt)

    for idx, stage in enumerate(stages):
        current_model, stage_history = _apply_stage(stage=stage,current_model=current_model,teacher=teacher,train_dataset=train_dataset,val_dataset=val_dataset,
                                                    config=config,domain=domain,pipeline_dir=pipeline_dir,pipeline_name=pipeline_name,
                                                    stage_idx=idx,stages=stages,)
        if stage_history is not None:
            all_stage_histories[f'stage{idx}_{stage}'] = serialize_history(
                stage_history
            )

    print("[PTQ] Applying post-training quantization — final step...")
    calibration_dataset = get_calibration_batches(train_dataset, config)
    tflite_path = os.path.join(
        pipeline_dir, f'{pipeline_name}_final.tflite'
    )
    tflite_bytes, _ = apply_ptq(current_model, calibration_dataset, tflite_path)

    print("[Eval] Evaluating float model...")
    float_metrics = evaluate_keras_model(current_model, test_dataset)
    print("[Eval] Evaluating TFLite model...")
    tflite_metrics = evaluate_tflite_model(tflite_bytes, test_dataset)
    print("[Eval] Computing accuracy drop vs baseline...")
    float_drop  = compute_accuracy_drop(baseline_metrics, float_metrics)
    tflite_drop = compute_accuracy_drop(baseline_metrics, tflite_metrics)
    print("[Eval] Computing L1 sensitivity profile...")
    l1_sensitivity = compute_l1_sensitivity(current_model, domain)
    print("[EI] Profiling on hardware targets...")
    hardware_profile = profile_tflite(tflite_path)
    print("[Eval] Assessing deployability...")
    deployability = assess_deployability(hardware_profile, hardware_config)

    result = {
        'pipeline': pipeline_name,'domain': domain,'seed': seed,'stages': stages,'float_metrics': float_metrics,'tflite_metrics': tflite_metrics,
        'float_drop': float_drop,'tflite_drop': tflite_drop,'l1_sensitivity': l1_sensitivity,'hardware_profile': hardware_profile,'deployability': deployability,
        'tflite_size_kb': round(os.path.getsize(tflite_path) / 1024, 3),'stage_histories':  all_stage_histories,
        'config_snapshot': {
            'kd_learning_rate': config['kd_learning_rate'],'kd_epochs': config['kd_epochs'],'qat_epochs': config['qat_epochs'],'prune_ratio': config['prune_ratio'],
            'qat_patience': config.get('qat_patience', 10),'finetune_lr': config['finetune_lr'],'finetune_epochs': config['finetune_epochs'],
        },
    }

    print(f"\n [Summary]")
    print(f"Float acc: {float_metrics['accuracy']:.4f} | F1: {float_metrics['macro_f1']:.4f}")
    print(f"TFLite acc: {tflite_metrics['accuracy']:.4f} | F1: {tflite_metrics['macro_f1']:.4f}")
    print(f"Acc drop: {tflite_drop['accuracy_drop']:.4f}")
    print(f"Size (KB): {result['tflite_size_kb']}")
    for mcu, d in deployability.items():
        print(
            f"{mcu}: RAM {d['ram_used_kb']}KB / {d['ram_budget_kb']}KB | "f"ROM {d['rom_used_kb']}KB / {d['rom_budget_kb']}KB | "f"Deployable: {d['deployable']}"
        )

    save_result(result, eval_dir, pipeline_name)

    del current_model
    keras.backend.clear_session()
    gc.collect()

    return result


def run_all_pipelines(domain, pipelines, base_config, domains_config, hardware_config, teacher_path, pipeline_dir, eval_dir, seed,):
    config = get_merged_config(base_config, domains_config[domain])

    print(f"\n[Setup] Loading data — domain={domain}, seed={seed}...")
    train_dataset, val_dataset, test_dataset = load_data(domain, config, seed)

    print(f"[Setup] Loading teacher from {teacher_path}...")
    teacher = load_keras_model(teacher_path)
    teacher.trainable = False

    print(f"[Setup] Loading student baseline metrics...")
    baseline_eval_path = os.path.join(eval_dir, 'baseline.json')

    if os.path.exists(baseline_eval_path):
        # Baseline already run — load from disk
        with open(baseline_eval_path) as f:
            baseline_result = json.load(f)
        student_baseline_metrics = baseline_result['float_metrics']
        print(
            f"  [Loaded] Student baseline accuracy : "
            f"{student_baseline_metrics['accuracy']:.4f} | "
            f"F1 : {student_baseline_metrics['macro_f1']:.4f}"
        )
    else:
        # Baseline not yet run find it in the pipelines list and run it
        baseline_pipeline_list = [p for p in pipelines if p['name'] == 'baseline']
        if not baseline_pipeline_list:
            raise RuntimeError(
                "Baseline result not found on disk and 'baseline' pipeline not in the current run list."
            )
        baseline_result = run_pipeline(pipeline=baseline_pipeline_list[0],domain=domain,config=config,hardware_config=hardware_config,teacher=teacher,
            baseline_metrics=evaluate_keras_model(teacher, test_dataset),train_dataset=train_dataset,val_dataset=val_dataset,test_dataset=test_dataset,
            pipeline_dir=pipeline_dir,eval_dir=eval_dir,seed=seed,)
        student_baseline_metrics = baseline_result['float_metrics']
        print(
            f"Student baseline accuracy : {student_baseline_metrics['accuracy']:.4f} | " f"F1 :{student_baseline_metrics['macro_f1']:.4f}"
        )

    all_results = []

    for pipeline in pipelines:
        if pipeline['name'] == 'baseline':
            continue
        result = run_pipeline(pipeline=pipeline,domain=domain,config=config,hardware_config=hardware_config,teacher=teacher, baseline_metrics=student_baseline_metrics,
            train_dataset=train_dataset,val_dataset=val_dataset,test_dataset=test_dataset,pipeline_dir=pipeline_dir,eval_dir=eval_dir,seed=seed,)
        if result:
            all_results.append(result)

    print(f"\n[Analysis] Computing Pareto frontier across all pipelines...")
    pareto = compute_pareto_frontier(all_results)
    pareto_path = os.path.join(eval_dir, 'pareto_frontier.json')
    with open(pareto_path, 'w') as f:
        json.dump(pareto, f, indent=2)
    print(f"Pareto frontier saved → {pareto_path}")
    print(f"Pareto-optimal pipelines: {[p['pipeline'] for p in pareto]}")

    del teacher
    keras.backend.clear_session()
    gc.collect()

    print(f"\n[Done] All pipelines complete domain={domain}, seed={seed}.")
    return all_results