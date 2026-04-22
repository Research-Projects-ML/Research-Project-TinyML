# experiments/metrics.py

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix


def evaluate_keras_model(model, dataset):
    all_preds  = []
    all_labels = []

    for inputs, labels in dataset:
        logits      = model(inputs, training=False)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32).numpy()
        all_preds.extend(predictions.tolist())
        all_labels.extend(labels.numpy().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy   = np.mean(all_preds == all_labels)
    macro_f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()
    conf_matrix  = confusion_matrix(all_labels, all_preds).tolist()

    return {
        'accuracy':      round(float(accuracy), 4),
        'macro_f1':      round(float(macro_f1), 4),
        'per_class_f1':  [round(f, 4) for f in per_class_f1],
        'confusion_matrix': conf_matrix,
    }


def evaluate_tflite_model(tflite_model_bytes, dataset):
    interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale      = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]

    all_preds  = []
    all_labels = []

    for inputs, labels in dataset:
        inputs_np = inputs.numpy()
        for i in range(len(inputs_np)):
            sample    = inputs_np[i:i+1].astype(np.float32)
            quantized = np.round(
                sample / input_scale + input_zero_point
            ).astype(np.int8)

            interpreter.set_tensor(input_details[0]['index'], quantized)
            interpreter.invoke()
            output     = interpreter.get_tensor(output_details[0]['index'])
            prediction = int(np.argmax(output, axis=-1)[0])

            all_preds.append(prediction)
            all_labels.append(int(labels[i].numpy()))

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy     = np.mean(all_preds == all_labels)
    macro_f1     = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()
    conf_matrix  = confusion_matrix(all_labels, all_preds).tolist()

    return {
        'accuracy':      round(float(accuracy), 4),
        'macro_f1':      round(float(macro_f1), 4),
        'per_class_f1':  [round(f, 4) for f in per_class_f1],
        'confusion_matrix': conf_matrix,
    }


def compute_accuracy_drop(baseline_metrics, compressed_metrics):
    return {
        'accuracy_drop': round(
            baseline_metrics['accuracy'] - compressed_metrics['accuracy'], 4
        ),
        'macro_f1_drop': round(
            baseline_metrics['macro_f1'] - compressed_metrics['macro_f1'], 4
        ),
    }


def compute_l1_sensitivity(model, domain):
    from compression.pruning import compute_channel_importance
    from tensorflow.keras import layers
    from models.timeseries.tcn import CausalDilatedConv1D

    sensitivity = {}

    for layer in model.layers:
        if isinstance(layer, (layers.Conv2D, layers.Conv1D)):
            importance = compute_channel_importance(layer)
            sensitivity[layer.name] = {
                'mean_importance': round(float(np.mean(importance)), 6),
                'std_importance':  round(float(np.std(importance)), 6),
                'cv':              round(
                    float(np.std(importance) / (np.mean(importance) + 1e-8)), 6
                ),
            }
        elif isinstance(layer, CausalDilatedConv1D):
            importance = compute_channel_importance(layer.conv)
            sensitivity[layer.name] = {
                'mean_importance': round(float(np.mean(importance)), 6),
                'std_importance':  round(float(np.std(importance)), 6),
                'cv':              round(
                    float(np.std(importance) / (np.mean(importance) + 1e-8)), 6
                ),
            }

    return sensitivity


def compute_accuracy_drop_sensitivity(model, dataset, domain, prune_ratio=0.3):
    from compression.pruning import (
        compute_channel_importance,
        get_channels_to_keep,
        prune_conv2d,
        prune_conv1d,
    )
    from tensorflow.keras import layers
    from models.timeseries.tcn import CausalDilatedConv1D

    baseline_metrics = evaluate_keras_model(model, dataset)
    baseline_acc     = baseline_metrics['accuracy']

    prunable_layers = []
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            prunable_layers.append(('conv2d', layer))
        elif isinstance(layer, layers.Conv1D) and layer.name.endswith('_proj') is False:
            prunable_layers.append(('conv1d', layer))
        elif isinstance(layer, CausalDilatedConv1D):
            prunable_layers.append(('causal', layer))

    sensitivity = {}

    for layer_type, layer in prunable_layers:
        weights_backup = layer.get_weights()

        try:
            if layer_type == 'conv2d':
                importance   = compute_channel_importance(layer)
                keep_indices = get_channels_to_keep(importance, prune_ratio)
                zeroed       = layer.get_weights()[0].copy()
                all_indices  = np.arange(zeroed.shape[-1])
                drop_indices = np.setdiff1d(all_indices, keep_indices)
                zeroed[:, :, :, drop_indices] = 0.0
                layer.set_weights([zeroed])

            elif layer_type in ('conv1d', 'causal'):
                inner = layer.conv if layer_type == 'causal' else layer
                importance   = compute_channel_importance(inner)
                keep_indices = get_channels_to_keep(importance, prune_ratio)
                zeroed       = inner.get_weights()[0].copy()
                all_indices  = np.arange(zeroed.shape[-1])
                drop_indices = np.setdiff1d(all_indices, keep_indices)
                zeroed[:, :, drop_indices] = 0.0
                inner.set_weights([zeroed])

            perturbed_metrics = evaluate_keras_model(model, dataset)
            acc_drop = round(baseline_acc - perturbed_metrics['accuracy'], 4)

        except Exception as e:
            acc_drop = None
            print(f"  [Sensitivity] Skipping {layer.name}: {e}")

        finally:
            layer.set_weights(weights_backup)

        layer_key = layer.name
        sensitivity[layer_key] = {
            'accuracy_drop': acc_drop,
        }

    return {
        'baseline_accuracy': round(baseline_acc, 4),
        'per_layer':         sensitivity,
    }


def serialize_history(history_dict):
    return {
        k: [round(float(v), 4) for v in vals]
        for k, vals in history_dict.items()
    }