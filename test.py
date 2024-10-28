import cv2
import numpy as np
import tensorflow as tf
TF_ENABLE_ONEDNN_OPTS=0

# Parameters
IMG_SIZE = (224, 224)  # Resize images to this size
MODEL_PATH = 'object_detection_model.keras'  # Path to the saved model

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Function to draw bounding box on the image
def draw_bounding_box(image, box):
    xmin, ymin, xmax, ymax = map(int, box)  # Convert to integer
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Draw rectangle in green
    return image

# Start video capture from camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera; adjust if necessary

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()  # Capture frame-by-frame

    if not ret:
        print("Error: Could not read frame.")
        break

    # Prepare the image for prediction
    input_image = cv2.resize(frame, IMG_SIZE)
    input_image = np.expand_dims(input_image, axis=0) / 255.0  # Normalize and add batch dimension

    # Predict bounding box
    predictions = model.predict(input_image)
    print(f"Predicted box: {predictions[0]}")  # Show the predictions in the console

    # Draw the bounding box on the original frame
    frame_with_box = draw_bounding_box(frame, predictions[0])

    # Show the image with bounding box
    cv2.imshow('Real-Time Object Detection', frame_with_box)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
