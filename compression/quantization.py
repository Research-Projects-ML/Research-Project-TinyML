import numpy as np
import tensorflow as tf


def get_representative_dataset(calibration_dataset, num_samples=200):
    count = 0
    for inputs, _ in calibration_dataset:
        inputs_np = inputs.numpy()
        for i in range(len(inputs_np)):
            if count >= num_samples:
                return
            sample = inputs_np[i:i+1]
            yield [tf.cast(tf.constant(sample), tf.float32)]
            count += 1


def apply_ptq(model, calibration_dataset, model_save_path):
    sample_count = sum(1 for _ in get_representative_dataset(calibration_dataset, num_samples=200))
    assert sample_count > 0, (
        "Representative dataset generator yielded no samples. "
        "Ensure calibration_dataset is a fresh tf.data slice that has not been exhausted."
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: get_representative_dataset(
        calibration_dataset, num_samples=200
    )
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_dtype  = interpreter.get_input_details()[0]['dtype']
    output_dtype = interpreter.get_output_details()[0]['dtype']
    assert input_dtype  == np.int8, (
        f"Input dtype is {input_dtype}, expected int8. "
        "Check representative dataset and converter config."
    )
    assert output_dtype == np.int8, (
        f"Output dtype is {output_dtype}, expected int8. "
        "Check representative dataset and converter config."
    )

    with open(model_save_path, 'wb') as f:
        f.write(tflite_model)

    return tflite_model, model_save_path


def dequantize_to_float(tflite_model_path, float_checkpoint_before_ptq):
    print(
        f"[Dequantize] Returning float checkpoint saved before PTQ. "
        f"TFLite reference: {tflite_model_path}"
    )
    return float_checkpoint_before_ptq


def requantize(float_model, calibration_dataset, model_save_path):
    return apply_ptq(float_model, calibration_dataset, model_save_path)