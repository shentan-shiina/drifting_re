#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e
echo "Colab-based Installation!"
echo "Creating virtual environment with uv..."
uv venv drifting_re --python 3.12

echo "Activating the environment..."
source .venv/bin/activate

echo "Installing frozen dependencies..."
uv pip install --python .venv/bin/python -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

echo "Installing the repo package in editable mode..."
uv pip install --python .venv/bin/python -e . 

echo "Setup complete! Run 'source .venv/bin/activate' to start using your environment."