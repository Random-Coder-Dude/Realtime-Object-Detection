import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Disable oneDNN optimizations (optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Parameters
IMG_SIZE = (224, 224)  # Resize images to this size
BATCH_SIZE = 32
EPOCHS = 100
DATA_DIR_IMAGES = 'data/images'  # Directory where images are stored
DATA_DIR_LABELS = 'data/labels'  # Directory where labels are stored

# Load images and bounding boxes from files
def load_data(data_dir_images, data_dir_labels):
    images = []
    bboxes = []
    
    for filename in os.listdir(data_dir_images):
        if filename.endswith(".jpg"):  # Adjust if using other formats
            img_path = os.path.join(data_dir_images, filename)
            image = cv2.imread(img_path)
            image = cv2.resize(image, IMG_SIZE)
            images.append(image)

            # Load corresponding bounding box for each image
            bbox_path = os.path.join(data_dir_labels, filename.replace('.jpg', '.txt'))
            if os.path.exists(bbox_path):
                with open(bbox_path, 'r') as f:
                    bbox = list(map(float, f.read().strip().split()))  # [xmin, ymin, xmax, ymax]
                    bboxes.append(bbox)
            else:
                bboxes.append([0, 0, 0, 0])  # Placeholder if no bbox found

    images = np.array(images)
    bboxes = np.array(bboxes)

    # Debugging: Check shapes
    print(f"Loaded {len(images)} images with shape {images.shape}")
    print(f"Loaded {len(bboxes)} bounding boxes with shape {bboxes.shape}")

    return images, bboxes

# Build a simple CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4)  # Output layer for bounding box (xmin, ymin, xmax, ymax)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Main function to train the model
def train_model():
    # Load data
    images, bboxes = load_data(DATA_DIR_IMAGES, DATA_DIR_LABELS)

    # Normalize images
    images = images.astype('float32') / 255.0

    # Set up training and validation data
    train_size = int(0.8 * len(images))  # 80% for training
    val_images = images[train_size:]
    val_bboxes = bboxes[train_size:]

    # Debugging: Check validation shapes
    print(f"Training with {train_size} images, validation with {len(val_images)} images")

    # Build and train the model
    model = build_model()

    # Fit the model without early stopping
    model.fit(images[:train_size], bboxes[:train_size], 
              epochs=EPOCHS, 
              validation_data=(val_images, val_bboxes))

    # Save the model in Keras format
    model.save('object_detection_model.keras')

if __name__ == "__main__":
    train_model()
