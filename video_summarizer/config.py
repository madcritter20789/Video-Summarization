"""
Configuration file for Video Summarization project
Based on 2024-2025 best practices for customizable pipelines
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

# Video Processing Configuration
VIDEO_CONFIG = {
    # Frame extraction
    "frame_rate": 1,  # Frames per second to extract
    "frame_quality": 95,  # JPEG quality (0-100)
    "max_frames": None,  # Maximum frames to extract (None = unlimited)

    # Audio extraction
    "audio_format": "mp3",
    "audio_quality": "0",  # 0 = best quality

    # Video formats supported
    "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]
}

# Transcription Configuration
TRANSCRIPTION_CONFIG = {
    # Whisper model size: tiny, base, small, medium, large
    "model_size": "base",

    # Language (None for auto-detection)
    "language": None,

    # Task: transcribe or translate
    "task": "transcribe",

    # Enable word-level timestamps
    "word_timestamps": True,
}

# Summarization Configuration
SUMMARIZATION_CONFIG = {
    # Model for summarization
    "model_name": "facebook/bart-large-cnn",

    # Summary parameters
    "max_length": 150,
    "min_length": 30,
    "do_sample": False,

    # Chunk size for long transcripts
    "chunk_size": 1024,

    # Generate key moments with timestamps
    "extract_key_moments": True,
    "num_key_moments": 5,
}

# Vectorization Configuration
VECTORIZATION_CONFIG = {
    # Video frame model
    "video_model": "resnet50",

    # Text embedding model
    "text_model": "sentence-transformers/all-MiniLM-L6-v2",

    # Embedding dimension (set automatically by models)
    "video_dim": 2048,  # ResNet50 output
    "text_dim": 384,    # all-MiniLM-L6-v2 output
}

# Model Training Configuration
TRAINING_CONFIG = {
    # LSTM parameters
    "lstm_hidden_size": 256,
    "lstm_num_layers": 2,
    "lstm_epochs": 50,
    "lstm_batch_size": 16,
    "lstm_learning_rate": 0.001,

    # Transformer parameters
    "transformer_model": "t5-small",
    "transformer_epochs": 3,
    "transformer_batch_size": 4,
    "transformer_learning_rate": 5e-5,
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    # ROUGE metrics to compute
    "rouge_metrics": ["rouge1", "rouge2", "rougeL"],

    # Use stemmer for ROUGE
    "use_stemmer": True,

    # Plot configuration
    "save_plots": True,
    "plot_format": "png",
    "plot_dpi": 300,
}

# Logging Configuration
LOGGING_CONFIG = {
    # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    "level": "INFO",

    # Log format
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",

    # Log to file
    "log_to_file": True,
    "log_file": str(BASE_DIR / "logs" / "video_summarization.log"),

    # Log to console
    "log_to_console": True,
}

# Caching Configuration
CACHE_CONFIG = {
    # Enable caching for expensive operations
    "enable_cache": True,

    # Cache directory
    "cache_dir": str(BASE_DIR / ".cache"),

    # Cache models
    "cache_models": True,

    # Cache intermediate results
    "cache_intermediates": True,

    # Cache expiration (in days)
    "cache_expiry_days": 7,
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    # Use GPU if available
    "use_gpu": True,

    # Number of worker threads
    "num_workers": 4,

    # Batch processing
    "batch_size": 1,

    # Memory limits (in MB, None = unlimited)
    "max_memory_mb": None,
}

# Streamlit UI Configuration
STREAMLIT_CONFIG = {
    # UI theme
    "theme": "light",

    # Page configuration
    "page_title": "Video Summarization AI",
    "page_icon": "🎬",
    "layout": "wide",

    # Upload limits
    "max_upload_size_mb": 200,

    # Display options
    "show_progress": True,
    "show_timestamps": True,
    "show_keywords": True,
    "enable_download": True,

    # Auto-play videos
    "autoplay_video": False,
}

def get_config():
    """Get all configuration as a dictionary"""
    return {
        "video": VIDEO_CONFIG,
        "transcription": TRANSCRIPTION_CONFIG,
        "summarization": SUMMARIZATION_CONFIG,
        "vectorization": VECTORIZATION_CONFIG,
        "training": TRAINING_CONFIG,
        "evaluation": EVALUATION_CONFIG,
        "logging": LOGGING_CONFIG,
        "cache": CACHE_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "streamlit": STREAMLIT_CONFIG,
    }

def setup_logging():
    """Setup logging based on configuration"""
    import logging
    import sys

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, LOGGING_CONFIG["level"]))

    # Create formatter
    formatter = logging.Formatter(LOGGING_CONFIG["format"])

    # Console handler
    if LOGGING_CONFIG["log_to_console"]:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if LOGGING_CONFIG["log_to_file"]:
        log_file = Path(LOGGING_CONFIG["log_file"])
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def setup_directories():
    """Create all necessary directories"""
    directories = [
        DATA_DIR / "video",
        DATA_DIR / "audio",
        DATA_DIR / "frames",
        DATA_DIR / "transcripts",
        EMBEDDINGS_DIR / "video_vectors",
        EMBEDDINGS_DIR / "transcript_vectors",
        EMBEDDINGS_DIR / "combined_vectors",
        MODELS_DIR,
        RESULTS_DIR / "summaries" / "generated",
        Path(LOGGING_CONFIG["log_file"]).parent,
    ]

    if CACHE_CONFIG["enable_cache"]:
        directories.append(Path(CACHE_CONFIG["cache_dir"]))

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    return directories

if __name__ == "__main__":
    # Test configuration
    print("Video Summarization Configuration")
    print("=" * 50)

    config = get_config()
    for section, values in config.items():
        print(f"\n{section.upper()}:")
        for key, value in values.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 50)
    print("Setting up directories...")
    dirs = setup_directories()
    print(f"Created {len(dirs)} directories")

    print("\nSetting up logging...")
    logger = setup_logging()
    logger.info("Configuration loaded successfully!")
