#!/bin/bash

ENV_DIR="venv"

# Create the virtual environment if it does not exist
if [ ! -d "$ENV_DIR" ]; then
    echo "Virtual environment not found. Creating '$ENV_DIR'..."
    python3 -m venv "$ENV_DIR"
else
    echo "Virtual environment '$ENV_DIR' already exists."
fi

# Activate the virtual environment
echo "Activating the virtual environment..."
source "$ENV_DIR/bin/activate"

if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment activated: $VIRTUAL_ENV"
else
    echo "Error: virtual environment NOT activated!"
fi

# Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt file not found. No dependencies installed."
fi