#!/bin/bash
# Convenience script to run the UI with proper Python path

# Set PYTHONPATH to project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run Streamlit
streamlit run ui/streamlit_app.py
