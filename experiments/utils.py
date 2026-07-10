import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import math
import json
import numpy as np
import tensorflow as tf
import keras
from data_loaders.image_loader import load_image_data
from data_loaders.timeseries_loader import load_timeseries_data
from models.timeseries.tcn import CausalDilatedConv1D, LastTimestep


def set_seed(seed: int) -> None:
    """
    Sets all random seeds for reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data(domain: str, config: dict, seed: int):
    """
    Loads train, validation, and test datasets for the given domain.
    """
    if domain == 'timeseries':
        return load_timeseries_data(config, seed)
    elif domain == 'image':
        return load_image_data(config, seed)
    else:
        raise ValueError(
            f"Unknown domain: '{domain}'.'timeseries' or 'image'."
        )

def get_steps_per_epoch(dataset: tf.data.Dataset) -> int:
    cardinality = dataset.cardinality().numpy()
    return int(cardinality) if cardinality > 0 else sum(1 for _ in dataset)

def load_keras_model(path: str) -> keras.Model:
    """
    Loads a .keras model file.
    Registers CausalDilatedConv1D and LastTimestep as custom objects so timeseries TCN models deserialise correctly without requiring
    the caller to pass custom_objects every time.
    """
    return keras.models.load_model(
        path,
        custom_objects={
            'CausalDilatedConv1D': CausalDilatedConv1D,
            'LastTimestep': LastTimestep,
        }
    )

def save_keras_model(model: keras.Model, path: str) -> None:
    """
    Saves a Keras model to a .keras file. Creates parent directories if they do not exist.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"[Checkpoint] Saved to{path}")


def get_calibration_batches(train_dataset: tf.data.Dataset, config: dict):
    """
    Returns ceil(200 / batch_size) batches for PTQ calibration.
    Consistent with the main pipeline calibration sample count.
    Dataset iterators are stateful, re-call this function each time a fresh iterator is needed rather than reusing the same slice.
    """
    batches_needed = math.ceil(200 / config['batch_size'])
    return train_dataset.take(batches_needed)


def result_exists(directory: str, name: str) -> bool:
    """
    Returns True if a result JSON already exists for the given name. Used by resume logic to skip already-completed runs.
    """
    return os.path.exists(os.path.join(directory, f'{name}.json'))


def save_result(result: dict, directory: str, name: str) -> None:
    """
    Saves a result dict as a JSON file. Creates the directory if it does not exist.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[Result] Saved to {path}")