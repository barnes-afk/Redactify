# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WHISPER_MODEL_SIZE=base \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE_TYPE=int8

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
# ffmpeg and ffprobe are required for audio utility bleeping
# libgomp1 is required by ctranslate2 (faster-whisper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download and install the English spaCy model required by Microsoft Presidio NLP config
RUN python -m spacy download en_core_web_sm

# Warm up / pre-download the default faster-whisper 'base' model
# This makes container startup and first execution much faster
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

# Copy the rest of the application code
COPY . .

# Expose the API port
EXPOSE 8000

# Command to run the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
