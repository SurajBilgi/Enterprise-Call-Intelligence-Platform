#!/bin/bash
# Convenience script to run the API with proper Python path

# Set PYTHONPATH to project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the API
python -m api.main
