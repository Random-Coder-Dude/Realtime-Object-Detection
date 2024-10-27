import cv2
import json
import os

# Parameters
IMAGE_FOLDER = 'data/images'  # Folder where the images are stored
ANNOTATION_FILE = 'data/annotations.json'  # JSON file to save the annotations

# Initialize annotation data structure
annotations = {}

# Load existing annotations if available
if os.path.exists(ANNOTATION_FILE):
    with open(ANNOTATION_FILE, 'r') as file:
        annotations = json.load(file)

# Variables to track drawing state
drawing = False
start_point = None
end_point = None
current_image = None

# Mouse callback function to draw rectangles
def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing, current_image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
            temp_image = current_image.copy()
            cv2.rectangle(temp_image, start_point, end_point, (255, 0, 0), 2)
            cv2.imshow("Annotator", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        if start_point and end_point:
            cv2.rectangle(current_image, start_point, end_point, (255, 0, 0), 2)
            cv2.imshow("Annotator", current_image)
            save_annotation(start_point, end_point)

# Save bounding box annotation
def save_annotation(start, end):
    global annotations, image_name

    x_min, y_min = min(start[0], end[0]), min(start[1], end[1])
    x_max, y_max = max(start[0], end[0]), max(start[1], end[1])
    bbox = [x_min, y_min, x_max, y_max]

    if image_name not in annotations:
        annotations[image_name] = []
    annotations[image_name].append(bbox)
    print(f"Annotation saved for {image_name}: {bbox}")

# Save annotations to JSON file
def save_annotations_to_file():
    with open(ANNOTATION_FILE, 'w') as file:
        json.dump(annotations, file, indent=4)
    print("Annotations saved to", ANNOTATION_FILE)

# Annotate images in the folder
def annotate_images():
    global current_image, image_name

    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for image_name in image_files:
        image_path = os.path.join(IMAGE_FOLDER, image_name)
        current_image = cv2.imread(image_path)

        if image_name in annotations:
            for bbox in annotations[image_name]:
                cv2.rectangle(current_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

        cv2.imshow("Annotator", current_image)
        cv2.setMouseCallback("Annotator", draw_rectangle)

        print(f"Annotating {image_name}. Press 'c' to skip, 's' to save and exit, or 'q' to quit.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):  # Skip to next image
                break
            elif key == ord('s'):  # Save and exit
                save_annotations_to_file()
                cv2.destroyAllWindows()
                return
            elif key == ord('q'):  # Quit without saving
                cv2.destroyAllWindows()
                return

    save_annotations_to_file()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    annotate_images()
