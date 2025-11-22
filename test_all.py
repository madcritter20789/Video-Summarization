#!/usr/bin/env python3
"""
Comprehensive test script for Video Summarization project
Tests all components without requiring actual video files
"""

import os
import sys
import tempfile
import numpy as np

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def test_imports():
    """Test if all required packages can be imported"""
    print_header("Testing Package Imports")

    packages = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "cv2": "OpenCV",
        "whisper": "OpenAI Whisper",
        "transformers": "Hugging Face Transformers",
        "faiss": "FAISS",
        "numpy": "NumPy",
        "PIL": "Pillow",
        "moviepy.editor": "MoviePy",
        "rouge_score": "ROUGE Score",
        "sentence_transformers": "Sentence Transformers",
        "matplotlib": "Matplotlib",
    }

    failed = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name:30} ... OK")
        except ImportError as e:
            print(f"✗ {name:30} ... FAILED")
            failed.append((name, str(e)))

    if failed:
        print("\n⚠ Some packages failed to import:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        return False
    else:
        print("\n✓ All packages imported successfully!")
        return True

def test_video_summarizer_modules():
    """Test video_summarizer modules"""
    print_header("Testing video_summarizer Modules")

    sys.path.insert(0, '/home/user/Video-Summarization/video_summarizer')

    modules = [
        ("src.extract_frames", "Frame Extraction"),
        ("src.extract_audio", "Audio Extraction"),
        ("src.transcribe_audio", "Audio Transcription"),
        ("src.vectorize_video", "Video Vectorization"),
        ("src.vectorize_transcript", "Transcript Vectorization"),
        ("src.combine_embeddings", "Embedding Combination"),
        ("src.summarize", "Summary Generation"),
        ("src.train_model", "Model Training"),
    ]

    failed = []
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"✓ {description:30} ... OK")
        except Exception as e:
            print(f"✗ {description:30} ... FAILED: {str(e)[:50]}")
            failed.append((description, str(e)))

    if failed:
        print("\n⚠ Some modules failed to load")
        return False
    else:
        print("\n✓ All modules loaded successfully!")
        return True

def test_streamlit_modules():
    """Test Video Streamlit modules"""
    print_header("Testing Video Streamlit Modules")

    sys.path.insert(0, '/home/user/Video-Summarization/Video Streamlit')

    modules = [
        ("utils.downloader", "YouTube Downloader"),
        ("utils.audio_transcriber", "Audio Transcriber"),
        ("utils.content_analysis", "Content Analysis"),
    ]

    failed = []
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"✓ {description:30} ... OK")
        except Exception as e:
            print(f"✗ {description:30} ... FAILED: {str(e)[:50]}")
            failed.append((description, str(e)))

    if failed:
        print("\n⚠ Some modules failed to load")
        return False
    else:
        print("\n✓ All modules loaded successfully!")
        return True

def test_directories():
    """Test if all required directories exist"""
    print_header("Testing Directory Structure")

    base_path = "/home/user/Video-Summarization/video_summarizer"
    directories = [
        "data/video",
        "data/audio",
        "data/frames",
        "data/transcripts",
        "embeddings/video_vectors",
        "embeddings/transcript_vectors",
        "embeddings/combined_vectors",
        "models",
        "results/summaries",
        "results/summaries/generated",
    ]

    missing = []
    for directory in directories:
        full_path = os.path.join(base_path, directory)
        if os.path.exists(full_path):
            print(f"✓ {directory:40} ... EXISTS")
        else:
            print(f"✗ {directory:40} ... MISSING")
            missing.append(directory)

    if missing:
        print(f"\n⚠ {len(missing)} directories are missing")
        return False
    else:
        print("\n✓ All required directories exist!")
        return True

def test_system_dependencies():
    """Test system dependencies"""
    print_header("Testing System Dependencies")

    import subprocess

    commands = {
        "ffmpeg": "FFmpeg (required for audio extraction)",
        "python3": "Python 3",
    }

    missing = []
    for cmd, description in commands.items():
        try:
            result = subprocess.run([cmd, "-version"],
                                  capture_output=True,
                                  timeout=5)
            if result.returncode == 0:
                print(f"✓ {description:40} ... INSTALLED")
            else:
                print(f"✗ {description:40} ... ERROR")
                missing.append(description)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"✗ {description:40} ... NOT FOUND")
            missing.append(description)

    if missing:
        print(f"\n⚠ Some system dependencies are missing:")
        for dep in missing:
            print(f"  - {dep}")
        if "FFmpeg" in str(missing):
            print("\nTo install FFmpeg:")
            print("  - Ubuntu/Debian: sudo apt-get install ffmpeg")
            print("  - macOS: brew install ffmpeg")
            print("  - Windows: Download from https://ffmpeg.org/")
        return False
    else:
        print("\n✓ All system dependencies are installed!")
        return True

def test_ground_truth_exists():
    """Test if ground truth file exists"""
    print_header("Testing Ground Truth Data")

    gt_path = "/home/user/Video-Summarization/video_summarizer/results/summaries/ground_truth.json"

    if os.path.exists(gt_path):
        print(f"✓ Ground truth file exists")
        try:
            import json
            with open(gt_path, 'r') as f:
                data = json.load(f)
            print(f"✓ Ground truth contains {len(data)} entries")
            return True
        except Exception as e:
            print(f"✗ Error reading ground truth: {e}")
            return False
    else:
        print(f"⚠ Ground truth file not found")
        print(f"  Location: {gt_path}")
        print(f"  This is required for model training and evaluation")
        return False

def test_model_initialization():
    """Test if models can be initialized"""
    print_header("Testing Model Initialization")

    try:
        import whisper
        print("✓ Loading Whisper model (base)...", end=" ")
        model = whisper.load_model("base")
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        return False

    try:
        from transformers import pipeline
        print("✓ Loading BART model...", end=" ")
        summarizer = pipeline("summarization",
                             model="facebook/bart-large-cnn",
                             clean_up_tokenization_spaces=False)
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        return False

    try:
        from sentence_transformers import SentenceTransformer
        print("✓ Loading Sentence Transformer...", end=" ")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        return False

    print("\n✓ All models initialized successfully!")
    return True

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  VIDEO SUMMARIZATION - COMPREHENSIVE TEST SUITE")
    print("="*60)

    results = {}

    # Run tests
    results['imports'] = test_imports()
    results['video_summarizer'] = test_video_summarizer_modules()
    results['streamlit'] = test_streamlit_modules()
    results['directories'] = test_directories()
    results['system_deps'] = test_system_dependencies()
    results['ground_truth'] = test_ground_truth_exists()

    # Model tests (these download models, so optional)
    print("\n⚠ Model initialization tests will download models (>1GB)")
    response = input("Run model tests? (y/n): ").lower().strip()
    if response == 'y':
        results['models'] = test_model_initialization()
    else:
        print("Skipping model tests...")
        results['models'] = None

    # Print summary
    print_header("TEST SUMMARY")

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    total = len(results)

    print(f"\nTotal Tests: {total}")
    print(f"  ✓ Passed:  {passed}")
    print(f"  ✗ Failed:  {failed}")
    print(f"  ⊘ Skipped: {skipped}")

    if failed == 0 and passed > 0:
        print("\n🎉 All tests passed! The project is ready to use.")
        print("\nNext steps:")
        print("  1. Place videos in video_summarizer/data/video/")
        print("  2. Run: cd video_summarizer && python main.py")
        print("  3. Or run Streamlit: cd 'Video Streamlit' && streamlit run app.py")
        return 0
    elif failed > 0:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        return 1
    else:
        print("\n✓ Setup looks good!")
        return 0

if __name__ == "__main__":
    exit(main())
