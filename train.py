import tensorflow as tf
import numpy as np
import cv2
import os
import json

# Parameters
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_SAVE_PATH = 'object_detection_model.tf'
IMAGE_FOLDER = 'data/images'
ANNOTATION_FILE = 'data/annotations.json'

# Load data from annotations JSON
def load_data():
    with open(ANNOTATION_FILE, 'r') as file:
        annotations = json.load(file)

    images = []
    bboxes = []
    for image_name, bbox_list in annotations.items():
        image_path = os.path.join(IMAGE_FOLDER, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.resize(image, IMAGE_SIZE)
        images.append(image)
        for bbox in bbox_list:
            x_min, y_min, x_max, y_max = bbox
            bbox_scaled = [
                x_min * IMAGE_SIZE[0] / image.shape[1],
                y_min * IMAGE_SIZE[1] / image.shape[0],
                x_max * IMAGE_SIZE[0] / image.shape[1],
                y_max * IMAGE_SIZE[1] / image.shape[0]
            ]
            bboxes.append(bbox_scaled)

    images = np.array(images, dtype='float32') / 255.0
    bboxes = np.array(bboxes, dtype='float32')
    return images, bboxes

# Define a CNN model for bounding box prediction
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Training function
def train_model():
    images, bboxes = load_data()
    model = create_model()
    model.fit(images, bboxes, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved at {MODEL_SAVE_PATH}")

# Convert to TensorFlow Lite with quantization
def convert_to_tflite():
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    def representative_data_gen():
        for _ in range(100):
            yield [tf.random.uniform([1, *IMAGE_SIZE, 3], 0, 1, dtype=tf.float32)]

    converter.representative_dataset = representative_data_gen
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()

    with open("object_detection_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("Model converted to object_detection_model.tflite")

# Compile for Coral USB Accelerator
def compile_for_edgetpu():
    os.system("edgetpu_compiler object_detection_model.tflite")
    print("Model compiled for Edge TPU as object_detection_model_edgetpu.tflite")

if __name__ == "__main__":
    train_model()
    convert_to_tflite()
    compile_for_edgetpu()
