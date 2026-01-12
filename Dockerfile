# Chatterbox TTS RunPod Worker
# GPU version for RunPod Serverless

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV TORCH_HOME=/models/torch

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Chatterbox TTS first (it will install its dependencies)
RUN pip install chatterbox-tts

# Force reinstall PyTorch stack with correct versions (AFTER chatterbox)
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install RunPod SDK
RUN pip install runpod==1.7.0 requests==2.31.0

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
