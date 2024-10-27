import cv2
import numpy as np
import os
import tensorflow as tf

try:
    from pycoral.adapters import common
    from pycoral.utils.edgetpu import make_interpreter
    CORAL_AVAILABLE = True
except ImportError:
    CORAL_AVAILABLE = False

# Parameters
MODEL_PATH_COMPUTER = 'object_detection_model.tflite'
MODEL_PATH_CORAL = 'object_detection_model_edgetpu.tflite'
IMAGE_SIZE = (128, 128)

# Load the model
def load_model():
    if CORAL_AVAILABLE:
        interpreter = make_interpreter(MODEL_PATH_CORAL)
        interpreter.allocate_tensors()
        print("Using Coral USB Accelerator.")
    else:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH_COMPUTER)
        interpreter.allocate_tensors()
        print("Using standard TensorFlow Lite interpreter.")
    return interpreter

# Draw bounding box on the image
def draw_bounding_box(image, bbox):
    x_min, y_min, x_max, y_max = map(int, bbox)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

# Perform inference and display results
def predict_and_display(interpreter, frame):
    input_data = cv2.resize(frame, IMAGE_SIZE)
    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB).astype(np.uint8)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], [input_data])
    interpreter.invoke()

    bbox = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]

    height, width, _ = frame.shape
    bbox[0] = int(bbox[0] * width / IMAGE_SIZE[0])
    bbox[1] = int(bbox[1] * height / IMAGE_SIZE[1])
    bbox[2] = int(bbox[2] * width / IMAGE_SIZE[0])
    bbox[3] = int(bbox[3] * height / IMAGE_SIZE[1])

    draw_bounding_box(frame, bbox)
    cv2.imshow("Real-Time Object Detection", frame)

def test_model():
    interpreter = load_model()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        predict_and_display(interpreter, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_model()
