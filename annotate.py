import cv2
import numpy as np
import os
TF_ENABLE_ONEDNN_OPTS=0

# Path to save annotated images and labels
image_folder = "data/images/"
label_folder = "data/labels/"
os.makedirs(label_folder, exist_ok=True)

drawing = False
ix, iy = -1, -1
annotations = []

# Mouse callback function to draw bounding boxes
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, annotations

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_image = img.copy()
            cv2.rectangle(temp_image, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Annotate", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        annotations.append((ix, iy, x, y))

# Load images and annotate
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = cv2.imread(os.path.join(image_folder, filename))
        cv2.imshow("Annotate", img)
        cv2.setMouseCallback("Annotate", draw_rectangle)

        # Press 'c' to save annotation, 'n' for next image
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                label_path = os.path.join(label_folder, filename.split('.')[0] + ".txt")
                np.savetxt(label_path, annotations, fmt='%d')
                annotations = []
                break
            elif key == ord('n'):
                annotations = []
                break

cv2.destroyAllWindows()
