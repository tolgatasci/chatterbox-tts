#!/usr/bin/env python3
"""
Pre-download Chatterbox models during Docker build.
This ensures models are baked into the image for fast cold starts.
"""

import os
import sys

# Set environment for model caching
os.environ["HF_HOME"] = "/models"
os.environ["TORCH_HOME"] = "/models/torch"

def download_models():
    print("=" * 60)
    print("Downloading Chatterbox TTS models...")
    print("=" * 60)

    try:
        import torch

        # Check CUDA availability
        if torch.cuda.is_available():
            device = "cuda"
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("CUDA not available, using CPU for download")

        # Download Standard model
        print("\n[1/1] Downloading Chatterbox Standard model (500M)...")
        print("      - Supports all languages via text input")
        print("      - Emotion control (exaggeration, cfg_weight)")
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device="cpu")
        del model
        print("✓ Standard model downloaded successfully!")

        # Summary
        print("\n" + "=" * 60)
        print("Model downloaded successfully!")
        print("=" * 60)
        print("\nModels cached in: /models")

        # List downloaded files
        total_size = 0
        for root, dirs, files in os.walk("/models"):
            for file in files:
                filepath = os.path.join(root, file)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                total_size += size_mb
                if size_mb > 10:
                    print(f"  {filepath} ({size_mb:.1f} MB)")

        print(f"\nTotal size: {total_size:.1f} MB")
        print("=" * 60)

    except Exception as e:
        print(f"Error downloading models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    download_models()
