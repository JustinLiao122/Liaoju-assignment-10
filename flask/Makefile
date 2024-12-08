# Variables
PYTHON = python3
VENV_DIR = .venv
MAIN_SCRIPT = app.py
REQUIREMENTS = requirements.txt

# Install dependencies and set up the virtual environment
install:
	python3 -m venv $(VENV_DIR)  # Create the virtual environment in the specified directory
	@echo "Installing PyTorch and torchvision..."
	./$(VENV_DIR)/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	@echo "Installing open-clip-torch..."
	./$(VENV_DIR)/bin/pip install open-clip-torch
	@echo "Installing requirments..."
	./$(VENV_DIR)/bin/pip install -r requirements.txt 
# Run the main script
run: 
	@echo "Running the application..."
	$(VENV_DIR)/bin/python3 $(MAIN_SCRIPT)

# Clean up the environment
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR)
	rm -rf __pycache__
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "Clean up complete."
