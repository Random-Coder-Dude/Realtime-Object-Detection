# Variables
MODEL_DIR=object_detection_model_saved_model.keras
TFLITE_MODEL=object_detection_model.tflite
PI_IP=192.168.1.2  # Replace with your Pi's IP
PI_USER=pi
PI_PATH=/home/pi/object_detection

# Train model
train:
	py training_pipeline.py

# Convert model to TFLite
convert:
	py -m tensorflow.lite.TFLiteConverter --saved_model_dir $(MODEL_DIR) --output_file $(TFLITE_MODEL)

# Transfer to Raspberry Pi
transfer:
	scp $(TFLITE_MODEL) $(PI_USER)@$(PI_IP):$(PI_PATH)

# Run testing script
deploy:
	ssh $(PI_USER)@$(PI_IP) 'python3 $(PI_PATH)/real_time_test.py'
