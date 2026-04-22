import tensorflow as tf
import tensorflow_datasets as tfds

def _cutout(image, mask_size=16):
    h, w = 32, 32
    half = mask_size // 2

    cx = tf.random.uniform(shape=[], minval=0, maxval=w, dtype=tf.int32)
    cy = tf.random.uniform(shape=[], minval=0, maxval=h, dtype=tf.int32)
    x1 = tf.maximum(0,  cx - half)
    x2 = tf.minimum(w,  cx + half)
    y1 = tf.maximum(0,  cy - half)
    y2 = tf.minimum(h,  cy + half)

    mask = tf.ones([h, w, 1], dtype=tf.float32)
    patch_h = y2 - y1
    patch_w = x2 - x1
    zero_patch = tf.zeros([patch_h, patch_w, 1], dtype=tf.float32)
    paddings = [[y1, h - y2], [x1, w - x2], [0, 0]]
    zero_padded = tf.pad(zero_patch, paddings, constant_values=1.0)

    mask = mask * zero_padded
    return image * mask


def load_image_data(config, seed):
    batch_size = config['batch_size']

    ds_train_full = tfds.load('cifar10', split='train', as_supervised=True, shuffle_files=True)
    ds_test       = tfds.load('cifar10', split='test',  as_supervised=True)

    num_train_total = 50000
    num_val         = 5000
    num_train       = num_train_total - num_val

    ds_train_full = ds_train_full.shuffle(buffer_size=num_train_total, seed=seed, reshuffle_each_iteration=False)
    ds_train_raw  = ds_train_full.take(num_train)
    ds_val_raw    = ds_train_full.skip(num_train)

    def normalise(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.pad_to_bounding_box(image, 4, 4, 40, 40)
        image = tf.image.random_crop(image, size=[32, 32, 3])
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    train_dataset = (
        ds_train_raw
        .map(normalise, num_parallel_calls=tf.data.AUTOTUNE)
        .map(augment,   num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=num_train, seed=seed)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (
        ds_val_raw
        .map(normalise, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_dataset = (
        ds_test
        .map(normalise, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, val_dataset, test_dataset