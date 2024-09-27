# Define virtual environment directory and Flask app entry point
VENV = venv
FLASK_APP = app.py

# Install dependencies within the virtual environment
install:
	python3 -m venv $(VENV)  # Create a virtual environment
	./$(VENV)/bin/pip install --upgrade pip  # Upgrade pip inside the venv
	./$(VENV)/bin/pip install -r requirements.txt  # Install required dependencies

# Run the Flask application using the virtual environment
run:
	FLASK_APP=$(FLASK_APP) FLASK_ENV=development ./$(VENV)/bin/flask run --port 3000

# Clean up virtual environment
clean:
	rm -rf $(VENV)

# Reinstall all dependencies (clean + install)
reinstall: clean install
