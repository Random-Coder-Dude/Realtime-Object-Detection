import cv2
import os
import yaml
import random
from shutil import copy2

# Load paths from custom YAML file
yaml_path = "data.yaml"  # Path to your custom YAML file

with open(yaml_path, "r") as file:
    data_config = yaml.safe_load(file)

# Directories from the YAML file
images_dir = "data/images"  # Root directory for images
train_images_dir = data_config["train"]
val_images_dir = data_config["val"]
train_labels_dir = data_config["labels"]["train"]
val_labels_dir = data_config["labels"]["val"]

# Ensure directories exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Variables to store bounding box coordinates and class ID
bbox = []
class_id = 0  # As specified in YAML, only 1 class (class_id = 0)

def draw_bbox(event, x, y, flags, param):
    global bbox, class_id

    # Left mouse button press: start drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox = [(x, y)]
    
    # Mouse movement while holding button: update current box
    elif event == cv2.EVENT_MOUSEMOVE:
        if len(bbox) == 1:
            img_copy = img.copy()
            cv2.rectangle(img_copy, bbox[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Image", img_copy)
    
    # Left mouse button release: finish drawing
    elif event == cv2.EVENT_LBUTTONUP:
        bbox.append((x, y))
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]
        
        # Get normalized YOLO format coordinates
        x_center = ((x1 + x2) / 2) / img.shape[1]
        y_center = ((y1 + y2) / 2) / img.shape[0]
        width = abs(x2 - x1) / img.shape[1]
        height = abs(y2 - y1) / img.shape[0]

        # Determine if image is for training or validation and save accordingly
        if img_name in train_images:
            label_path = os.path.join(train_labels_dir, f"{img_name}.txt")
        else:
            label_path = os.path.join(val_labels_dir, f"{img_name}.txt")
        
        # Save annotation in YOLO format
        with open(label_path, "a") as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        print(f"Saved annotation for {label_path}: class {class_id}, bbox {x_center:.4f}, {y_center:.4f}, {width:.4f}, {height:.4f}")
        
        # Reset bounding box for the next annotation
        bbox.clear()

# Split images 80/20 into train and validation
all_images = [img for img in os.listdir(images_dir) if img.endswith(".jpg") or img.endswith(".png")]
random.shuffle(all_images)

# Calculate split index ensuring 80% for training
split_idx = int(len(all_images) * 0.8)

# Ensure at least one image goes to both train and validation
if len(all_images) <= 1:
    train_images = all_images
    val_images = []
else:
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

# Move images to appropriate directories and annotate
for img_name in all_images:
    # Load image
    img_path = os.path.join(images_dir, img_name)
    img = cv2.imread(img_path)
    img_name_no_ext = os.path.splitext(img_name)[0]  # Remove file extension

    # Determine if image is for training or validation and move accordingly
    if img_name in train_images:
        dest_dir = train_images_dir
    else:
        dest_dir = val_images_dir
    
    copy2(img_path, dest_dir)

    # Set up window and callback
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_bbox)

    print(f"Draw bounding boxes for {img_name}. Press 'q' to move to the next image.")
    
    while True:
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

print("Annotation and split process completed.")
