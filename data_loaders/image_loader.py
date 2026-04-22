import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_image_data(config, seed):
    batch_size = config['batch_size']

    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train_full = x_train_full.astype('float32') / 255.0
    x_test       = x_test.astype('float32') / 255.0
    y_train_full = y_train_full.squeeze().astype('int32')
    y_test       = y_test.squeeze().astype('int32')

    num_train_total = len(x_train_full)
    num_val         = 5000
    num_train       = num_train_total - num_val

    rng             = np.random.default_rng(seed)
    shuffle_indices = rng.permutation(num_train_total)

    x_train_full = x_train_full[shuffle_indices]
    y_train_full = y_train_full[shuffle_indices]

    x_train, y_train = x_train_full[:num_train], y_train_full[:num_train]
    x_val,   y_val   = x_train_full[num_train:], y_train_full[num_train:]

    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.pad_to_bounding_box(image, 4, 4, 40, 40)
        image = tf.image.random_crop(image, size=[32, 32, 3])
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=num_train, seed=seed)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_dataset = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, val_dataset, test_dataset