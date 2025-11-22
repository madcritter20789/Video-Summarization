import whisper
import os
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading
_whisper_model_cache = {}

def get_whisper_model(model_name="base"):
    """
    Get cached Whisper model or load if not cached.

    Args:
        model_name: Whisper model size (tiny, base, small, medium, large)

    Returns:
        Loaded Whisper model
    """
    if model_name not in _whisper_model_cache:
        logger.info(f"Loading Whisper model '{model_name}'...")
        _whisper_model_cache[model_name] = whisper.load_model(model_name)
        logger.info(f"Whisper model '{model_name}' loaded successfully")
    return _whisper_model_cache[model_name]

def transcribe_audio(audio_path, transcript_path, model_name="base", language=None):
    """
    Transcribe audio file using Whisper with progress tracking.

    Args:
        audio_path: Path to audio file
        transcript_path: Path to save transcript
        model_name: Whisper model size (default: base)
        language: Language code (None for auto-detection)

    Returns:
        Dictionary with transcript and metadata
    """
    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)

    logger.info(f"Transcribing audio: {audio_path}")
    model = get_whisper_model(model_name)

    # Transcribe with progress callback
    print(f"Transcribing audio with Whisper ({model_name} model)...")

    transcribe_options = {
        "verbose": False,
        "language": language
    }

    result = model.transcribe(audio_path, **transcribe_options)

    # Save transcript
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(result['text'])

    logger.info(f"Transcription saved to {transcript_path}")
    logger.info(f"Detected language: {result.get('language', 'unknown')}")

    # Return full result including segments for timestamp extraction
    return {
        'text': result['text'],
        'language': result.get('language'),
        'segments': result.get('segments', [])
    }
