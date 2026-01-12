# Chatterbox TTS RunPod Worker
# GPU version for RunPod Serverless

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV TORCH_HOME=/models/torch

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install compatible torchaudio
RUN pip install torchaudio==2.4.0

# Install Chatterbox TTS
RUN pip install chatterbox-tts

# Install RunPod SDK and other dependencies
RUN pip install \
    runpod==1.7.0 \
    requests==2.31.0

# Create models directory
RUN mkdir -p /models

# Pre-download models during build
COPY download_models.py /app/download_models.py
RUN python /app/download_models.py

# Copy handler
COPY handler.py /app/handler.py

WORKDIR /app

# RunPod handler
CMD ["python", "-u", "handler.py"]
