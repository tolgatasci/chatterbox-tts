#!/usr/bin/env python3
"""
Chatterbox TTS RunPod Serverless Handler
Full-featured with multilingual support and voice cloning.

Models:
- Standard: English, emotion control (exaggeration, cfg_weight)
- Multilingual: 23+ languages
- Turbo: Fast English only

Supports both base64 and URL for reference audio.
"""

import os
import io
import base64
import tempfile
import runpod
import torch
import torchaudio
import requests

# Set environment
os.environ["HF_HOME"] = "/models"
os.environ["TORCH_HOME"] = "/models/torch"

# Supported languages for multilingual model
SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese"
}

# Global model instances (loaded once)
_standard_model = None
_multilingual_model = None
_turbo_model = None


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_standard_model():
    """Load Chatterbox Standard model (English, emotion control)."""
    global _standard_model
    if _standard_model is None:
        print("Loading Chatterbox Standard model (500M)...")
        from chatterbox.tts import ChatterboxTTS
        _standard_model = ChatterboxTTS.from_pretrained(device=get_device())
        print("Standard model loaded!")
    return _standard_model


def load_multilingual_model():
    """Load Chatterbox Multilingual model (23+ languages)."""
    global _multilingual_model
    if _multilingual_model is None:
        print("Loading Chatterbox Multilingual model (500M)...")
        from chatterbox.tts import ChatterboxTTS
        # Multilingual model uses same class but different weights
        _multilingual_model = ChatterboxTTS.from_pretrained(
            device=get_device(),
            model_id="ResembleAI/chatterbox-multilingual"
        )
        print("Multilingual model loaded!")
    return _multilingual_model


def load_turbo_model():
    """Load Chatterbox Turbo model (fast, English only)."""
    global _turbo_model
    if _turbo_model is None:
        try:
            print("Loading Chatterbox Turbo model (350M)...")
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            _turbo_model = ChatterboxTurboTTS.from_pretrained(device=get_device())
            print("Turbo model loaded!")
        except Exception as e:
            print(f"Turbo model not available ({e}), will use Standard")
            _turbo_model = None
    return _turbo_model


def base64_to_audio_file(audio_b64: str) -> str:
    """Convert base64 audio to a temporary file path."""
    # Remove data URL prefix if present
    if "," in audio_b64:
        audio_b64 = audio_b64.split(",")[1]

    # Decode base64
    audio_bytes = base64.b64decode(audio_b64)

    # Detect format from magic bytes
    suffix = ".wav"
    if audio_bytes[:4] == b"RIFF":
        suffix = ".wav"
    elif audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        suffix = ".mp3"
    elif audio_bytes[:4] == b"fLaC":
        suffix = ".flac"
    elif audio_bytes[:4] == b"OggS":
        suffix = ".ogg"

    # Write to temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(audio_bytes)
    temp_file.close()

    return temp_file.name


def url_to_audio_file(audio_url: str) -> str:
    """Download audio from URL to a temporary file path."""
    print(f"Downloading audio from: {audio_url}")

    response = requests.get(audio_url, timeout=60)
    response.raise_for_status()

    # Get content type to determine format
    content_type = response.headers.get("content-type", "").lower()
    if "wav" in content_type or "wave" in content_type:
        suffix = ".wav"
    elif "mp3" in content_type or "mpeg" in content_type:
        suffix = ".mp3"
    elif "flac" in content_type:
        suffix = ".flac"
    elif "ogg" in content_type:
        suffix = ".ogg"
    else:
        # Try to detect from URL
        if audio_url.endswith(".mp3"):
            suffix = ".mp3"
        elif audio_url.endswith(".flac"):
            suffix = ".flac"
        elif audio_url.endswith(".ogg"):
            suffix = ".ogg"
        else:
            suffix = ".wav"

    # Write to temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(response.content)
    temp_file.close()

    print(f"Downloaded {len(response.content)} bytes to {temp_file.name}")
    return temp_file.name


def audio_tensor_to_base64(audio_tensor: torch.Tensor, sample_rate: int = 24000,
                           output_format: str = "wav") -> str:
    """
    Convert audio tensor to base64 string.
    Supports wav, mp3, flac, ogg formats.
    """
    # Ensure tensor is on CPU and has correct shape
    audio = audio_tensor.cpu()

    # torchaudio expects (channels, samples) shape
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # Add channel dimension
    elif audio.dim() == 3:
        audio = audio.squeeze(0)  # Remove batch dimension if present

    # Ensure float32 and normalize to [-1, 1]
    audio = audio.float()
    if audio.abs().max() > 1.0:
        audio = audio / audio.abs().max()

    # Determine format
    fmt = output_format.lower().strip(".")
    if fmt not in ["wav", "mp3", "flac", "ogg"]:
        fmt = "wav"

    # Write to buffer
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio, sample_rate, format=fmt)
    buffer.seek(0)

    # Encode to base64
    audio_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return audio_b64


def handler(job):
    """
    RunPod serverless handler for Chatterbox TTS.

    Input parameters:
    ─────────────────
    Required:
    - text (str): Text to synthesize

    Voice Cloning (optional):
    - reference_audio_base64 (str): Base64 encoded reference audio
    - reference_audio_url (str): URL to download reference audio
      Note: base64 takes priority if both provided

    Model Selection:
    - model_type (str): "standard" | "multilingual" | "turbo" (default: "standard")
    - language (str): Language code for multilingual model (default: "en")
      Supported: ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh

    Generation Parameters:
    - exaggeration (float): Emotion intensity 0.0-2.0 (default: 1.0)
    - cfg_weight (float): CFG weight 0.0-1.0 (default: 0.5)
    - temperature (float): Sampling temperature 0.0-1.5 (default: 0.7)
    - top_p (float): Top-p sampling 0.0-1.0 (default: 0.9)
    - top_k (int): Top-k sampling (default: 50)
    - repetition_penalty (float): Repetition penalty (default: 1.0)
    - seed (int): Random seed for reproducibility (default: random)

    Output Options:
    - output_format (str): "wav" | "mp3" | "flac" | "ogg" (default: "wav")
    - sample_rate (int): Output sample rate (default: 24000)

    Returns:
    ────────
    - audio_base64 (str): Generated audio as base64
    - sample_rate (int): Audio sample rate
    - duration_seconds (float): Audio duration
    - model_type (str): Model used
    - language (str): Language used (if multilingual)
    - text_length (int): Input text length
    """
    try:
        job_input = job.get("input", {})

        # ─── Required: text to synthesize ───
        text = job_input.get("text", "").strip()
        if not text:
            return {"error": "Missing required parameter: text"}

        # ─── Model selection ───
        model_type = job_input.get("model_type", "standard").lower()
        language = job_input.get("language", "en").lower()

        # Validate language for multilingual
        if model_type == "multilingual" and language not in SUPPORTED_LANGUAGES:
            return {
                "error": f"Unsupported language: {language}",
                "supported_languages": list(SUPPORTED_LANGUAGES.keys())
            }

        # ─── Voice cloning reference audio ───
        reference_audio_b64 = job_input.get("reference_audio_base64")
        reference_audio_url = job_input.get("reference_audio_url")
        audio_prompt_path = None

        if reference_audio_b64:
            print("Converting reference audio from base64...")
            audio_prompt_path = base64_to_audio_file(reference_audio_b64)
            print(f"Reference audio saved to: {audio_prompt_path}")
        elif reference_audio_url:
            print("Downloading reference audio from URL...")
            audio_prompt_path = url_to_audio_file(reference_audio_url)
            print(f"Reference audio downloaded to: {audio_prompt_path}")

        # ─── Generation parameters ───
        exaggeration = float(job_input.get("exaggeration", 1.0))
        cfg_weight = float(job_input.get("cfg_weight", 0.5))
        temperature = float(job_input.get("temperature", 0.7))
        top_p = float(job_input.get("top_p", 0.9))
        top_k = int(job_input.get("top_k", 50))
        repetition_penalty = float(job_input.get("repetition_penalty", 1.0))
        seed = job_input.get("seed")

        # ─── Output options ───
        output_format = job_input.get("output_format", "wav").lower()
        output_sample_rate = int(job_input.get("sample_rate", 24000))

        # ─── Set seed if provided ───
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed(int(seed))

        # ─── Logging ───
        print(f"Generating speech...")
        print(f"  Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"  Model: {model_type}")
        print(f"  Language: {language} ({SUPPORTED_LANGUAGES.get(language, 'Unknown')})")
        print(f"  Voice cloning: {'Yes' if audio_prompt_path else 'No'}")
        print(f"  Exaggeration: {exaggeration}, CFG: {cfg_weight}")

        # ─── Generate audio based on model type ───
        if model_type == "turbo":
            model = load_turbo_model()
            if model is None:
                # Fallback to standard if turbo not available
                model = load_standard_model()
                model_type = "standard"
            wav = model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
            )

        elif model_type == "multilingual":
            model = load_multilingual_model()
            wav = model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                language_id=language,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )

        else:  # standard
            model = load_standard_model()
            wav = model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )

        print("Speech generated successfully!")

        # ─── Convert to output format ───
        native_sample_rate = 24000
        audio_b64 = audio_tensor_to_base64(wav, native_sample_rate, output_format)

        # ─── Calculate duration ───
        if wav.dim() == 1:
            num_samples = wav.shape[0]
        else:
            num_samples = wav.shape[-1]
        duration = num_samples / native_sample_rate

        print(f"Audio duration: {duration:.2f} seconds")
        print(f"Output format: {output_format}")

        # ─── Cleanup temp file ───
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            os.remove(audio_prompt_path)

        # ─── Return response ───
        response = {
            "audio_base64": audio_b64,
            "sample_rate": native_sample_rate,
            "duration_seconds": round(duration, 2),
            "model_type": model_type,
            "text_length": len(text),
            "output_format": output_format,
        }

        # Add language if multilingual
        if model_type == "multilingual":
            response["language"] = language
            response["language_name"] = SUPPORTED_LANGUAGES.get(language)

        return response

    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error: {error_msg}")
        print(traceback_str)
        return {
            "error": error_msg,
            "traceback": traceback_str
        }


# ─── Health check endpoint ───
def health_check(_):
    """Health check for RunPod."""
    return {
        "status": "healthy",
        "models": ["standard", "multilingual", "turbo"],
        "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
        "device": get_device()
    }


# Start RunPod serverless handler
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True
})
