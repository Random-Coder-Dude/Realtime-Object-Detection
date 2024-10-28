import tensorflow as tf
TF_ENABLE_ONEDNN_OPTS=0

# Load the Keras model
model = tf.keras.models.load_model('object_detection_model.keras')

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("object_detection_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully converted to TFLite format.")
