import cv2
from ultralytics import YOLO

# Load the YOLO model (replace 'yolo11n.pt' with your trained model path)
model = YOLO("yolo11n.pt")

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam; use another number if you have multiple cameras

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Real-time detection loop
try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Run YOLO model on the frame
        results = model(frame)[0]  # Directly access the first (and only) result

        # Extract and draw bounding boxes, class labels, and confidence scores
        for result in results.boxes:  # Access boxes directly
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
            confidence = result.conf[0]  # Confidence score
            class_id = int(result.cls[0])  # Class ID

            # Convert class ID to class name
            class_name = model.names[class_id]

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{class_name} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Display the resulting frame
        cv2.imshow("Real-Time YOLO Detection", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Real-time detection stopped by user.")

finally:
    # Release the capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
