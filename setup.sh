#!/bin/bash

# Video Summarization Setup Script
# This script sets up the environment for running the Video Summarization project locally

set -e  # Exit on error

echo "========================================="
echo "Video Summarization - Local Setup"
echo "========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "Python version: $(python3 --version)"
echo ""

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Warning: ffmpeg is not installed."
    echo "ffmpeg is required for audio extraction from videos."
    echo "Please install ffmpeg using:"
    echo "  - Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  - macOS: brew install ffmpeg"
    echo "  - Windows: Download from https://ffmpeg.org/download.html"
    echo ""
    read -p "Do you want to continue without ffmpeg? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "ffmpeg is installed: $(ffmpeg -version | head -n 1)"
fi
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install video_summarizer dependencies
echo ""
echo "Installing video_summarizer dependencies..."
cd video_summarizer
pip install -r requirements.txt

# Install Video Streamlit dependencies
echo ""
echo "Installing Video Streamlit dependencies..."
cd "../Video Streamlit"
pip install -r requirements.txt

# Go back to root directory
cd ..

# Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p video_summarizer/data/video
mkdir -p video_summarizer/data/audio
mkdir -p video_summarizer/data/frames
mkdir -p video_summarizer/data/transcripts
mkdir -p video_summarizer/embeddings/video_vectors
mkdir -p video_summarizer/embeddings/transcript_vectors
mkdir -p video_summarizer/embeddings/combined_vectors
mkdir -p video_summarizer/models
mkdir -p video_summarizer/results/summaries/generated
mkdir -p "Video Streamlit/static/uploaded_videos"

echo ""
echo "========================================="
echo "Setup completed successfully!"
echo "========================================="
echo ""
echo "To use the project:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. For video processing: cd video_summarizer && python main.py"
echo "3. For Streamlit frontend: cd 'Video Streamlit' && streamlit run app.py"
echo ""
echo "Note: Make sure to place your videos in video_summarizer/data/video/"
echo ""
