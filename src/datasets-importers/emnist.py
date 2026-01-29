import tensorflow_datasets as tfds
import numpy as np

(ds_train, ds_test), ds_info = tfds.load(
    "emnist/digits",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

def tfds_to_numpy(ds):
    xs, ys = [], []
    for x, y in ds:
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.array(xs), np.array(ys)

x_train, y_train = tfds_to_numpy(ds_train)
x_test, y_test = tfds_to_numpy(ds_test)