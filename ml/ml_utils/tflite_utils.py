import tensorflow as tf
import numpy as np
from pathlib import Path


def export_to_tflite(keras_model, out_dir="models", rep_data=None) -> Path:
    """
    Quantise `keras_model` to full-int8 and save.
    rep_data – optional N×seq×feat ndarray for representative dataset.
    Returns the Path to the `.tflite` file.
    """
    if rep_data is None:
        rep_data = np.random.rand(100, *keras_model.input_shape[1:]).astype("float32")

    def rep_gen():
        for i in range(len(rep_data)):
            yield [rep_data[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite = converter.convert()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (keras_model.name + "_int8.tflite")
    out_path.write_bytes(tflite)
    return out_path

