import tensorflow as tf

# Load the trained Keras model
MODEL_PATH = 'object_detection_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Convert the model to a TensorFlow Lite format with integer quantization
def convert_to_tflite():
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization and quantization options
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Define representative dataset for quantization
    def representative_data_gen():
        for _ in range(100):  # Use a small sample of images for calibration
            # Randomly generate data within expected input range (assuming 128x128)
            yield [tf.random.uniform([1, 128, 128, 3], 0, 1, dtype=tf.float32)]
    
    converter.representative_dataset = representative_data_gen
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # Convert model to TensorFlow Lite format
    tflite_model = converter.convert()
    
    # Save the converted model to file
    with open("object_detection_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("Model converted and saved as object_detection_model.tflite")

if __name__ == "__main__":
    convert_to_tflite()
