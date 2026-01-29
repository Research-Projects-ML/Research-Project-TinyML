# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import tensorflow_datasets as tfds

from models.tabular import mlp
from models.audio import cnn_ds
from models.timeseries import cnn_1d

from compression.quantization import quantize_model
from src.compression.distilation import Distiller

SCENARIO = "timeseries"
# DISTILLATION = True
DISTILLATION = False
QUANTIZATION = True
# QUANTIZATION = False

x_train = None
y_train = None
x_test = None
y_test = None
model = None

if SCENARIO == "tabular":

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
    model = mlp.build()

elif SCENARIO == "audio":

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    model = cnn_ds.build()

elif SCENARIO == "timeseries":

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784, 1).astype("float32") / 255.0
    model = cnn_1d.build()


# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")


if DISTILLATION:
    teacher = mlp.build()
    teacher.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    teacher.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
    student = mlp.build()
    distiller = Distiller(student, teacher, alpha=0.1, temperature=3.0)
    distiller.compile(
        optimizer="adam",
        metrics=["accuracy"],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
        distill_loss_fn=tf.keras.losses.KLDivergence()
    )
    distiller.fit(x_train, y_train, epochs=5, batch_size=64)
    model = student

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")


converter = tf.lite.TFLiteConverter.from_keras_model(model)
modelLight = converter.convert()

# Export models
model.export("tf/models/model")
with open("tflite/models/model.tflite", "wb") as f:
    f.write(modelLight)
if QUANTIZATION:
    modelLightQuantized = quantize_model(model)
    with open("tflite/models/modelQuantized.tflite", "wb") as f:
        f.write(modelLightQuantized)

# Export test data
np.save("tflite/data/x_test", x_test)
np.save("tflite/data/y_test", y_test)