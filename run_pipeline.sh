#!/bin/bash
# Convenience script to run the pipeline with proper Python path

# Set PYTHONPATH to project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the orchestrator
python pipelines/orchestrator.py "$@"
