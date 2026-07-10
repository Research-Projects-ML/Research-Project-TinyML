import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
import keras
from experiments.utils import get_steps_per_epoch


def get_representative_dataset(calibration_dataset, num_samples=200):
    """
    Yields individual float32 samples for PTQ calibration.
    The TFLite converter calls this to compute per-layer activation ranges for INT8 quantization.
    """
    representative_data = calibration_dataset.unbatch().take(num_samples)
    for sample, _ in representative_data:
        yield [tf.expand_dims(tf.cast(sample, tf.float32), axis=0)]


def apply_qat(model, train_dataset, val_dataset, config):
    """
    Reference: Quantization-Aware Fine-Tuning via INT8-scaled uniform noise injection, adapted from Baskin et al. (2021) UNIQ, motivated by Jacob et al. (2018).
    Injects uniform noise scaled to INT8 rounding error into each trainable weight tensor before every forward pass. The model learns to be robust
    to this perturbation, reducing PTQ accuracy loss at deployment.
    Noise per tensor: step = (2 * max(|w|)) / 255,  noise ~ Uniform(-step/2, step/2)
    Only trainable variables are perturbed: moving_mean and moving_variance are never touched. Validation is also run under noise so early stopping
    monitors quantization-aware performance, not float performance.
    Input: float32 Keras model (must be pre-trainedrandom weights will not converge).
    Output: (float32 model, history dict). PTQ must follow to produce the INT8 TFLite binary.
    """

    NUM_INT8_LEVELS = 255.0

    initial_lr = config['qat_learning_rate']
    epochs = config['qat_epochs']
    patience = config.get('qat_patience', 10)

    steps_per_epoch = get_steps_per_epoch(train_dataset)

    lr_schedule = keras.optimizers.schedules.CosineDecay(initial_learning_rate=initial_lr,decay_steps=epochs * steps_per_epoch,alpha=1e-6,)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def train_step(images, labels):
        original_weights = [tf.identity(v) for v in model.trainable_variables]

        for var in model.trainable_variables:
            w_range = tf.reduce_max(tf.abs(var))
            step = (2.0 * w_range) / NUM_INT8_LEVELS
            noise = tf.random.uniform(tf.shape(var), -step / 2.0, step / 2.0, dtype=var.dtype)
            var.assign_add(noise)

        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss = loss_fn(labels, logits)

        grads = tape.gradient(loss, model.trainable_variables)

        for var, orig in zip(model.trainable_variables, original_weights):
            var.assign(orig)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, logits

    @tf.function
    def val_step(images):
        return model(images, training=False)

    best_val_acc = -np.inf
    best_weights = None
    patience_counter = 0

    history = {
        'train_loss': [], 'train_accuracy': [],
        'val_loss': [], 'val_accuracy': [],
    }

    for epoch in range(epochs):
        train_losses  = []
        train_correct = 0
        train_samples = 0

        for images, labels in train_dataset:
            loss, logits = train_step(images, labels)
            train_losses.append(loss.numpy())
            preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
            train_correct += tf.reduce_sum(tf.cast(tf.equal(preds, tf.cast(labels, tf.int32)), tf.int32)).numpy()
            train_samples += len(labels)

        epoch_train_loss = float(np.mean(train_losses))
        epoch_train_acc = train_correct / train_samples

        # Inject noise into trainable variables for validation (moving_mean and moving_variance are not injected with noise)
        original_trainable = [tf.identity(v) for v in model.trainable_variables]

        for var in model.trainable_variables:
            w_range = tf.reduce_max(tf.abs(var))
            step = (2.0 * w_range) / NUM_INT8_LEVELS
            noise = tf.random.uniform(
                tf.shape(var), -step / 2.0, step / 2.0, dtype=var.dtype
            )
            var.assign_add(noise)

        val_losses = []
        val_correct = 0
        val_samples = 0

        for images, labels in val_dataset:
            logits = val_step(images)
            loss = loss_fn(labels, logits)
            preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
            val_losses.append(loss.numpy())
            val_correct += tf.reduce_sum(tf.cast(tf.equal(preds, tf.cast(labels, tf.int32)), tf.int32)).numpy()
            val_samples += len(labels)

        for var, orig in zip(model.trainable_variables, original_trainable):
            var.assign(orig)

        epoch_val_loss = float(np.mean(val_losses))
        epoch_val_acc = val_correct / val_samples

        history['train_loss'].append(epoch_train_loss)
        history['train_accuracy'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_acc)

        print(
            f"QAT Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}"
        )

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_weights = model.get_weights()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"QAT early stopping at epoch {epoch+1}. Best val acc: {best_val_acc:.4f}")
            break

    if best_weights is None:
        raise RuntimeError("[QAT] No epoch completed, check train_dataset is non-empty.")

    model.set_weights(best_weights)
    return model, history


def apply_ptq(model, calibration_dataset, model_save_path):
    """
    Converts a float32 Keras model to a fully INT8 TFLite binary for Cortex-M deployment.
    Uses 200 calibration samples to compute per-layer activation ranges, enforces TFLITE_BUILTINS_INT8 (no float fallback), 
    and asserts INT8 dtype on both input and output ends before saving.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: get_representative_dataset(
        calibration_dataset, num_samples=200
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_dtype = interpreter.get_input_details()[0]['dtype']
    output_dtype = interpreter.get_output_details()[0]['dtype']
    assert input_dtype == np.int8, (f"Input dtype is {input_dtype}, expected int8. Check representative dataset and converter config.")
    assert output_dtype == np.int8, (f"Output dtype is {output_dtype}, expected int8. Check representative dataset and converter config.")

    with open(model_save_path, 'wb') as f:
        f.write(tflite_model)

    return tflite_model, model_save_path