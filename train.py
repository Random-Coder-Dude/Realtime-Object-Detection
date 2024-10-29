#!Copied from documentation

from ultralytics import YOLO

# Load the model with your custom YAML configuration and optionally pretrained weights
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # Adjust path to .pt if necessary

# Set training parameters
results = model.train(
    data="data.yaml",  # Point to your custom data YAML
    epochs=100,                        # Set to desired number of epochs
    imgsz=640,                         # Image size for training (e.g., 640x640)
    batch=16,                          # Batch size (adjust as needed for GPU memory)
    workers=4                          # Number of data loading workers
)

# Print summary of results after training
print("Training complete!")
print("Results:", results)
