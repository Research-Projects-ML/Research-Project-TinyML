import os
import zipfile
import urllib.request
import numpy as np
import tensorflow as tf

UCI_HAR_URL = (
    'https://d396qusza40orc.cloudfront.net/getdata/projectfiles/UCI%20HAR%20Dataset.zip'
)
# 6 activity classes: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
NUM_CLASSES = 6
# Input shape: 128 timesteps, 9 sensor channels (accelerometer XYZ + gyroscope XYZ + body acc XYZ)
TIMESTEPS       = 128
INPUT_CHANNELS  = 9



def _download_and_extract(data_dir):
    extract_path = os.path.join(data_dir, 'UCI HAR Dataset')

    if os.path.exists(extract_path):
        print(f"[UCI HAR] Dataset already exists at {extract_path}")
        return extract_path

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, 'uci_har.zip')
    print("[UCI HAR Dataset] Downloading dataset...")
    urllib.request.urlretrieve(UCI_HAR_URL, zip_path)
    print("[UCI HAR Dataset] Extracting...")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(data_dir)
    os.remove(zip_path)
    print(f"[UCI HAR Dataset] is ready ... {extract_path}")
    return extract_path


def _load_signals(data_path, split):
    signal_names = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z'
    ]
    signals = []
    signals_path = os.path.join(data_path, split, 'Inertial Signals')

    for name in signal_names:
        file_path = os.path.join(signals_path, f'{name}_{split}.txt')
        signal = np.loadtxt(file_path)  # shape: (num_samples, 128)
        signals.append(signal)

    # Stack to (num_samples, 128, 9)
    signals = np.stack(signals, axis=-1).astype(np.float32)
    return signals


def _load_labels(data_path, split):
    label_path = os.path.join(data_path, split, f'y_{split}.txt')
    labels = np.loadtxt(label_path, dtype=np.int32)
    labels = labels - 1
    return labels


def _normalise_signals(train_signals, val_signals, test_signals):
    mean = train_signals.mean(axis=(0, 1), keepdims=True)  # (1, 1, 9)
    std  = train_signals.std(axis=(0, 1),  keepdims=True)  # (1, 1, 9)
    std  = np.where(std == 0, 1.0, std)  # avoid division by zero

    train_signals = (train_signals - mean) / std
    val_signals   = (val_signals   - mean) / std
    test_signals  = (test_signals  - mean) / std

    return train_signals, val_signals, test_signals


def load_timeseries_data(config, seed):
    batch_size = config['batch_size']

    data_dir = config.get(
        'data_dir',
        '/content/drive/MyDrive/tinyml_notebook/datasets'
    )

    data_path = _download_and_extract(data_dir)

    train_signals = _load_signals(data_path, 'train')  # (7352, 128, 9)
    train_labels  = _load_labels(data_path, 'train')   # (7352,)
    test_signals  = _load_signals(data_path, 'test')   # (2947, 128, 9)
    test_labels   = _load_labels(data_path, 'test')    # (2947,)

    SPLIT_SEED = 42  # fixed

    rng = np.random.default_rng(SPLIT_SEED)

    shuffle_indices = rng.permutation(len(train_signals))
    
    train_signals = train_signals[shuffle_indices]
    train_labels = train_labels[shuffle_indices]
    
    num_train_total = len(train_signals)
    num_val = int(num_train_total * 0.2)
    num_train = num_train_total - num_val
    
    val_signals = train_signals[num_train:]
    val_labels = train_labels[num_train:]
    train_signals = train_signals[:num_train]
    train_labels = train_labels[:num_train]

    train_signals, val_signals, test_signals = _normalise_signals(
        train_signals, val_signals, test_signals
    )

    def make_dataset(signals, labels, shuffle, batch_size, seed):
        ds = tf.data.Dataset.from_tensor_slices((signals, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(signals), seed=seed)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
    
    train_dataset = make_dataset(train_signals, train_labels, shuffle=True, batch_size=batch_size, seed=seed)
    val_dataset = make_dataset(val_signals, val_labels, shuffle=False, batch_size=batch_size, seed=seed)
    test_dataset = make_dataset(test_signals, test_labels, shuffle=False, batch_size=batch_size, seed=seed)

    return train_dataset, val_dataset, test_dataset