# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import tensorflow_datasets as tfds
from tensorflow.keras.models import clone_model

from models.tabular import mlp
from models.tabular import mlp2
from models.audio import cnn_ds
from models.timeseries import cnn_1d

from compression.quantization import quantize_model
from src.compression.distilation import Distiller

from data.loaders import human_activity_recognition, unsw

from utils.plot_f1 import plot_f1
from utils.metrics import F1Metric

SCENARIO = "timeseries"
DISTILLATION = True
# DISTILLATION = False
QUANTIZATION = True
# QUANTIZATION = False

x_train = None
y_train = None
x_test = None
y_test = None
model = None

if SCENARIO == "tabular":

    x_train, y_train, x_test, y_test = human_activity_recognition.load_data("data/uci-human-activity-recognition")
    model = mlp.build()

elif SCENARIO == "audio":

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    model = cnn_ds.build()

elif SCENARIO == "timeseries":

    x_train, y_train, x_test, y_test = unsw.load_data("data/UNSW-NB15")
    model = mlp2.build(input_dim=x_train.shape[1], num_classes=6)


# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', F1Metric()]
)
# Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
plot_f1(history, "normal.png")

test_loss, test_acc, f1 = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")


if DISTILLATION:
    teacher = clone_model(model)
    student = clone_model(model)
    teacher.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", F1Metric()]
    )
    history_teacher = teacher.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
    test_loss, test_acc, f1 = teacher.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    plot_f1(history_teacher, "teacher.png")
    distiller = Distiller(student, teacher, alpha=0.1, temperature=3.0)
    distiller.compile(
        optimizer="adam",
        metrics=["accuracy" ,F1Metric()],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
        distill_loss_fn=tf.keras.losses.KLDivergence()
    )
    distiller.fit(x_train, y_train, epochs=5, batch_size=64)
    model = student

    test_loss, test_acc, f1 = model.evaluate(x_test, y_test)
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