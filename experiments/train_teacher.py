import os
import sys 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config import CHECKPOINT_DIR
from experiments.utils import set_seed, get_steps_per_epoch
from experiments.pipeline import get_merged_config
from data_loaders.timeseries_loader import load_timeseries_data
from data_loaders.image_loader import load_image_data
from models.timeseries.tcn import get_teacher as get_ts_teacher
from models.image.resnet8 import get_teacher as get_image_teacher
from evaluation.model_metrics import evaluate_keras_model

import argparse
import numpy as np
import tensorflow as tf
import keras

TEACHER_SEED = 42

def train_ts_teacher(config, train_dataset, val_dataset):
    model = get_ts_teacher(config)
    steps_per_epoch = get_steps_per_epoch(train_dataset)
    epochs = config['train_epochs']

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config['train_lr'],
        decay_steps=epochs * steps_per_epoch,
        alpha=1e-6
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        jit_compile=False
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=config['early_stop_patience'],restore_best_weights=True,verbose=1),
        keras.callbacks.ModelCheckpoint(filepath=os.path.join(CHECKPOINT_DIR, 'tcn_teacher_best.keras'),monitor='val_accuracy',save_best_only=True,verbose=1),
    ]

    print(f"\n[TCN Teacher] Training for up to {epochs} epochs...")
    print(f"steps_per_epoch : {steps_per_epoch}")
    print(f"total decay_steps: {epochs * steps_per_epoch}")

    history = model.fit(train_dataset,validation_data=val_dataset,epochs=epochs,callbacks=callbacks,verbose=1)

    return model, history.history

def train_image_teacher(config, train_dataset, val_dataset):
    model = get_image_teacher(config)
    steps_per_epoch = get_steps_per_epoch(train_dataset)
    epochs = config['train_epochs']

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config['train_lr'],
        decay_steps=epochs * steps_per_epoch,
        alpha=1e-6
    )

    # Label smoothing of 0.1 helps ResNet-8 generalise on CIFAR-10 the teacher needs to produce soft, calibrated logits for KD to work well.
    # A teacher that is overconfident produces near-one-hot soft targets that give the student almost no information beyond the hard labels.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        jit_compile=False
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=config['early_stop_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, 'resnet8_teacher_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
    ]

    print(f"\n[ResNet-8 Teacher] Training for up to {epochs} epochs...")
    print(f"steps_per_epoch : {steps_per_epoch}")
    print(f"total decay_steps: {epochs * steps_per_epoch}")

    history = model.fit(train_dataset,validation_data=val_dataset,epochs=epochs,callbacks=callbacks,verbose=1)

    return model, history.history


def train_teacher(domain):
    print(f"\n{'='*55}")
    print(f"  Training teacher domain={domain} | seed={TEACHER_SEED}")
    print(f"{'='*55}")

    set_seed(TEACHER_SEED)

    config = get_merged_config(domain)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    name_map  = {'timeseries': 'tcn_teacher', 'image': 'resnet8_teacher'}
    save_path = os.path.join(CHECKPOINT_DIR, f"{name_map[domain]}.keras")

    if os.path.exists(save_path):
        print(f"\n[Skip] Teacher checkpoint already exists: {save_path}")
        print(f"  Delete it manually if you want to retrain.")
        return

    print(f"\n[Data] Loading {domain} data...")
    if domain == 'timeseries':
        train_dataset, val_dataset, test_dataset = load_timeseries_data(
            config, seed=TEACHER_SEED
        )
        model, history = train_ts_teacher(config, train_dataset, val_dataset)
        final_path = save_path
    elif domain == 'image':
        train_dataset, val_dataset, test_dataset = load_image_data(
            config, seed=TEACHER_SEED
        )
        model, history = train_image_teacher(config, train_dataset, val_dataset)
        final_path = save_path

    else:
        raise ValueError(f"Unknown domain: {domain}. Choose 'timeseries' or 'image'.")

    print(f"\n[Eval] Evaluating teacher on test set...")
    metrics = evaluate_keras_model(model, test_dataset)
    print(f"  Test accuracy : {metrics['accuracy']:.4f}")
    print(f"  Test macro F1 : {metrics['macro_f1']:.4f}")

    # Warn if teacher accuracy is too low to be a useful reference.
    min_acc = {'timeseries': 0.85, 'image': 0.70}
    if metrics['accuracy'] < min_acc[domain]:
        print(
            f"\n[Warning] Teacher accuracy {metrics['accuracy']:.4f} is below "
            f"the expected minimum of {min_acc[domain]:.2f} for {domain}. "
            f"Check training config before using this teacher in pipelines."
        )

    model.save(final_path)
    print(f"\n[Saved] Teacher checkpoint → {final_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TinyML teacher models.')
    parser.add_argument(
        '--domain',
        type=str,
        required=True,
        choices=['timeseries', 'image', 'both']
    )
    args = parser.parse_args()

    if args.domain == 'both':
        train_teacher('timeseries')
        train_teacher('image')
    else:
        train_teacher(args.domain)