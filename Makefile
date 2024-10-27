# Variables
MODEL_NAME=object_detection_model
MODEL_DIR=$(MODEL_NAME)_saved_model
MODEL_TFLITE=$(MODEL_NAME).tflite
MODEL_EDGETPU=$(MODEL_NAME)_edgetpu.tflite
RPI_USER=pi
RPI_HOST=raspberrypi.local
RPI_DIR=/home/$(RPI_USER)/model

# Python environment variables
PYTHON=py

# Paths
ANNOTATION_SCRIPT=annotate.py
TRAINING_SCRIPT=train.py
TEST_SCRIPT=real_time_test.py

# Commands
.PHONY: all train convert compile transfer deploy clean

# 1. Run the entire pipeline (train, convert, compile, transfer)
all: train convert compile transfer

# 2. Train the model and save as SavedModel format
train:
	@echo "Starting training..."
	$(PYTHON) $(TRAINING_SCRIPT) --save_format saved_model --output_dir $(MODEL_DIR)
	@echo "Training complete. Model saved in directory $(MODEL_DIR)"

# 3. Convert the SavedModel to TFLite
convert: $(MODEL_DIR)
	@echo "Converting model to TensorFlow Lite..."
	$(PYTHON) -c "import tensorflow as tf; \
	converter = tf.lite.TFLiteConverter.from_saved_model('$(MODEL_DIR)'); \
	converter.optimizations = [tf.lite.Optimize.DEFAULT]; \
	tflite_model = converter.convert(); \
	open('$(MODEL_TFLITE)', 'wb').write(tflite_model)"
	@echo "Conversion complete. Model saved as $(MODEL_TFLITE)"

# 4. Compile for Coral Edge TPU
compile: $(MODEL_TFLITE)
	@echo "Compiling for Edge TPU..."
	edgetpu_compiler $(MODEL_TFLITE)
	@echo "Compilation complete. Compiled model saved as $(MODEL_EDGETPU)"

# 5. Transfer model and scripts to Raspberry Pi
transfer:
	@echo "Transferring files to Raspberry Pi..."
	scp $(MODEL_EDGETPU) $(MODEL_TFLITE) $(TEST_SCRIPT) $(ANNOTATION_SCRIPT) \
		$(RPI_USER)@$(RPI_HOST):$(RPI_DIR)
	@echo "Transfer complete. Files moved to $(RPI_DIR) on Raspberry Pi."

# 6. Run testing on Raspberry Pi (connects via SSH)
deploy:
	@echo "Starting real-time testing on Raspberry Pi..."
	ssh $(RPI_USER)@$(RPI_HOST) 'cd $(RPI_DIR) && $(PYTHON) $(TEST_SCRIPT)'

# 7. Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	rm -rf $(MODEL_DIR) $(MODEL_TFLITE) $(MODEL_EDGETPU)
	@echo "Cleanup complete."
