# Enterprise Call Intelligence Platform - Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model (if needed)
# RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create storage directories
RUN mkdir -p storage/raw_transcripts \
    storage/structured \
    storage/vectors \
    storage/cache \
    storage/search \
    logs

# Expose ports
EXPOSE 8000 8501

# Default command (can be overridden)
CMD ["python", "-m", "api.main"]
