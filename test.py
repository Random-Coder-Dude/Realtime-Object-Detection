import cv2
from ultralytics import YOLO

#! Can Load any .pt file (Using pre trained model)
model = YOLO("yolo11n.pt")

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # Use 0 always

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.") #! Debug Call
    exit()

# Main Loop
try:
    # Infinite Loop
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.") #! Debug Call
            break

        # Run YOLO model on the frame
        results = model(frame)[0]  # #! [0] part just tells the first result, can be changed

        # Extract and draw bounding boxes, class labels, and confidence scores (all from documentation)
        for result in results.boxes:  # Access Output directly
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
            confidence = result.conf[0]  # Confidence score
            class_id = int(result.cls[0])  # Class ID (Irelevant)

            # Convert class ID to class name (Irelevant, 1 class)
            class_name = model.names[class_id]

            # Draw bounding box and label on the frame (ChatGPT helped)
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

        # Display the resulting frame, basicly acts like viewport
        cv2.imshow("Test", frame)

        # Break the loop on 'q' key press
        #! End infinite loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Also make sure ctrl+c works #! Reason for try: at the top
except KeyboardInterrupt:
    print("Real-time detection stopped by user.")

#! if else then structure
finally:
    # Boilerplate (keep)
    cap.release()
    cv2.destroyAllWindows()
