# Chatterbox TTS RunPod Worker

Full-featured voice cloning TTS worker using [Chatterbox](https://github.com/resemble-ai/chatterbox) by ResembleAI.

## Features

- 🎭 **Voice Cloning** - Clone any voice from 5-second audio sample
- 🌍 **23 Languages** - Multilingual support (Turkish, English, German, etc.)
- 😊 **Emotion Control** - Adjust exaggeration for expressive speech
- 🎵 **Multiple Formats** - Output as WAV, MP3, FLAC, or OGG
- ⚡ **Fast Cold Starts** - Models pre-downloaded (~6GB)

## Models

| Model | Size | Languages | Features |
|-------|------|-----------|----------|
| `standard` | 500M | English | Emotion control |
| `multilingual` | 500M | 23 languages | Multi-language |
| `turbo` | 350M | English | Fastest |

## Supported Languages

```
ar (Arabic)     da (Danish)     de (German)     el (Greek)
en (English)    es (Spanish)    fi (Finnish)    fr (French)
he (Hebrew)     hi (Hindi)      it (Italian)    ja (Japanese)
ko (Korean)     ms (Malay)      nl (Dutch)      no (Norwegian)
pl (Polish)     pt (Portuguese) ru (Russian)    sv (Swedish)
sw (Swahili)    tr (Turkish)    zh (Chinese)
```

## API Parameters

### Required
| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | string | Text to synthesize |

### Voice Cloning
| Parameter | Type | Description |
|-----------|------|-------------|
| `reference_audio_base64` | string | Base64 encoded audio for voice cloning |
| `reference_audio_url` | string | URL to download reference audio |

### Model Selection
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | string | "standard" | "standard", "multilingual", or "turbo" |
| `language` | string | "en" | Language code (for multilingual) |

### Generation Parameters
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `exaggeration` | float | 1.0 | 0.0-2.0 | Emotion intensity |
| `cfg_weight` | float | 0.5 | 0.0-1.0 | CFG weight |
| `temperature` | float | 0.7 | 0.0-1.5 | Sampling temperature |
| `top_p` | float | 0.9 | 0.0-1.0 | Top-p sampling |
| `top_k` | int | 50 | - | Top-k sampling |
| `repetition_penalty` | float | 1.0 | - | Repetition penalty |
| `seed` | int | random | - | Random seed for reproducibility |

### Output Options
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_format` | string | "wav" | "wav", "mp3", "flac", or "ogg" |

## Example Requests

### Simple TTS
```json
{
  "input": {
    "text": "Hello, this is a test.",
    "model_type": "standard"
  }
}
```

### Turkish with Voice Cloning
```json
{
  "input": {
    "text": "Merhaba, bu benim sesimle konuşuyor!",
    "model_type": "multilingual",
    "language": "tr",
    "reference_audio_url": "https://example.com/my_voice.wav",
    "exaggeration": 1.2
  }
}
```

### Expressive English
```json
{
  "input": {
    "text": "Wow! This is amazing news!",
    "model_type": "standard",
    "exaggeration": 1.8,
    "output_format": "mp3"
  }
}
```

### Full Parameters
```json
{
  "input": {
    "text": "Full control over every parameter.",
    "model_type": "multilingual",
    "language": "en",
    "reference_audio_base64": "UklGRi...",
    "exaggeration": 1.0,
    "cfg_weight": 0.5,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "seed": 42,
    "output_format": "wav"
  }
}
```

## Response

```json
{
  "audio_base64": "UklGRi...",
  "sample_rate": 24000,
  "duration_seconds": 3.5,
  "model_type": "multilingual",
  "language": "tr",
  "language_name": "Turkish",
  "text_length": 42,
  "output_format": "wav"
}
```

## Deploy to RunPod

### 1. Fork & Clone
```bash
git clone https://github.com/tolgatasci/chatterbox-tts.git
cd chatterbox-tts
```

### 2. Set GitHub Secrets
Go to repo Settings > Secrets > Actions:
- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: Your Docker Hub access token

### 3. Push to Build
```bash
git push origin main
```

GitHub Actions builds and pushes to: `docker.io/tolgatasci/chatterbox-tts:latest`

### 4. Create RunPod Endpoint
1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Image: `tolgatasci/chatterbox-tts:latest`
4. GPU: A40 or A100 recommended
5. Create!

## Local Testing

```bash
# Build CPU version
docker build -f Dockerfile.cpu -t chatterbox-tts:cpu .

# Run test
docker run --rm \
  -v $(pwd)/test_input.json:/app/test_input.json \
  chatterbox-tts:cpu
```

## Tips

- 🎤 **Voice Cloning**: Use 5-15 second clear audio samples
- 🌍 **Language Match**: Reference audio should match target language
- 😊 **Emotion**: Higher exaggeration = more expressive
- 📁 **File Size**: Keep reference audio under 5MB
- 🔊 **Quality**: WAV for best quality, MP3 for smaller size
