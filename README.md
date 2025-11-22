# Video Summarization Project

A comprehensive video summarization system that extracts, analyzes, and summarizes video content using machine learning and natural language processing techniques.

## Features

### Core Video Summarization (video_summarizer/)
- **Video Processing Pipeline**: Extract frames and audio from videos
- **Audio Transcription**: Transcribe audio using OpenAI Whisper
- **Frame Vectorization**: Convert video frames to embeddings using ResNet50
- **Transcript Vectorization**: Convert transcripts to embeddings using sentence transformers
- **Combined Embeddings**: Merge visual and textual features
- **Summary Generation**: Create detailed summaries using BART model
- **Model Training**: Train LSTM and Transformer models for custom summarization
- **Evaluation**: ROUGE score evaluation with visualization

### Frontend Interface (Video Streamlit/)
- **Streamlit Web App**: User-friendly interface for video analysis
- **YouTube Support**: Download and process YouTube videos
- **Video Upload**: Upload and process local video files
- **Content Analysis**: Get transcripts, summaries, and insights
- **Keyword Extraction**: Identify important keywords from content

## Prerequisites

### System Requirements
- Python 3.8 or higher
- ffmpeg (for audio extraction)
- 4GB+ RAM recommended
- GPU optional but recommended for faster processing

### Installing ffmpeg
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Installation

### Quick Setup
Run the automated setup script:
```bash
chmod +x setup.sh
./setup.sh
```

### Manual Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Video-Summarization
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
# Install video_summarizer dependencies
cd video_summarizer
pip install -r requirements.txt

# Install Video Streamlit dependencies
cd "../Video Streamlit"
pip install -r requirements.txt
```

4. **Create necessary directories**
```bash
cd ..
mkdir -p video_summarizer/data/{video,audio,frames,transcripts}
mkdir -p video_summarizer/embeddings/{video_vectors,transcript_vectors,combined_vectors}
mkdir -p video_summarizer/{models,results/summaries/generated}
mkdir -p "Video Streamlit/static/uploaded_videos"
```

## Usage

### 1. Video Processing Pipeline

Process videos through the complete pipeline:

```bash
cd video_summarizer
# Place your videos in data/video/
python main.py
```

This will:
- Extract frames from videos
- Extract and transcribe audio
- Generate video and transcript embeddings
- Create combined embeddings
- Generate detailed summaries
- Save results in `results/summaries/`

### 2. Streamlit Web Interface

Launch the interactive web application:

```bash
cd "Video Streamlit"
streamlit run app.py
```

Features:
- Upload video files (MP4, AVI, MOV)
- Enter YouTube URLs
- Get transcripts, summaries, and insights
- View keyword analysis

### 3. Model Training

Train custom models on your data:

```bash
cd video_summarizer
python src/train_model.py
```

Requirements:
- Combined embeddings in `embeddings/combined_vectors/`
- Ground truth summaries in `results/summaries/ground_truth.json`

### 4. Evaluation

Evaluate summarization performance:

```bash
cd video_summarizer
python evaluate.py
```

This will:
- Compare generated summaries with ground truth
- Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- Generate visualization plots
- Save results to `results/summaries/rouge_scores.png`

## Project Structure

```
Video-Summarization/
├── video_summarizer/
│   ├── src/
│   │   ├── extract_frames.py       # Frame extraction
│   │   ├── extract_audio.py        # Audio extraction
│   │   ├── transcribe_audio.py     # Audio transcription
│   │   ├── vectorize_video.py      # Frame vectorization
│   │   ├── vectorize_transcript.py # Transcript vectorization
│   │   ├── combine_embeddings.py   # Embedding combination
│   │   ├── summarize.py            # Summary generation
│   │   └── train_model.py          # Model training
│   ├── data/                       # Input data directories
│   ├── embeddings/                 # Vector embeddings
│   ├── models/                     # Trained models
│   ├── results/                    # Output summaries
│   ├── main.py                     # Main pipeline
│   ├── app.py                      # Streamlit app
│   ├── evaluate.py                 # Evaluation script
│   └── requirements.txt            # Dependencies
│
├── Video Streamlit/
│   ├── utils/
│   │   ├── downloader.py           # YouTube downloader
│   │   ├── audio_transcriber.py    # Audio transcription
│   │   └── content_analysis.py     # Content analysis
│   ├── static/uploaded_videos/     # Uploaded videos
│   ├── app.py                      # Streamlit interface
│   ├── main_tool.py                # Processing logic
│   └── requirements.txt            # Dependencies
│
├── setup.sh                        # Setup script
└── README.md                       # This file
```

## Dependencies

### Core Dependencies
- **torch**: Deep learning framework
- **transformers**: NLP models (BART, T5, Sentence Transformers)
- **openai-whisper**: Audio transcription
- **opencv-python**: Video frame extraction
- **faiss-cpu**: Vector similarity search
- **moviepy**: Video processing
- **ffmpeg-python**: Audio extraction
- **rouge-score**: Evaluation metrics
- **streamlit**: Web interface

### Optional Dependencies
- **matplotlib**: Plotting (evaluation)
- **pytube**: YouTube downloads
- **youtube-transcript-api**: YouTube transcripts

## Configuration

### Ground Truth Format
Create ground truth summaries in `video_summarizer/results/summaries/ground_truth.json`:

```json
[
  {
    "input": "Description of video content",
    "output": "Expected summary text"
  },
  {
    "input": "Another video description",
    "output": "Another expected summary"
  }
]
```

### Model Parameters
Edit model parameters in the respective source files:
- Frame rate: `src/extract_frames.py` (default: 1 FPS)
- Whisper model: `src/transcribe_audio.py` (default: "base")
- Summarization model: `src/summarize.py` (default: "facebook/bart-large-cnn")

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Install ffmpeg using system package manager
   - Ensure ffmpeg is in system PATH

2. **Out of memory errors**
   - Reduce frame extraction rate
   - Process videos in smaller batches
   - Use smaller model variants ("tiny" instead of "base" for Whisper)

3. **CUDA/GPU errors**
   - Install CPU versions: `pip install faiss-cpu`
   - Models will automatically fall back to CPU

4. **YouTube download errors**
   - Update pytube: `pip install --upgrade pytube`
   - Some videos may have download restrictions

5. **Streamlit port already in use**
   - Specify different port: `streamlit run app.py --server.port 8502`

## Performance Tips

- **GPU Acceleration**: Install CUDA-compatible PyTorch for faster processing
- **Batch Processing**: Process multiple videos in one run using `main.py`
- **Model Selection**: Use smaller models for faster inference (trade-off with accuracy)
- **Frame Rate**: Adjust frame extraction rate based on video content type

## Testing

To test all features:

1. **Test video processing**:
   ```bash
   cd video_summarizer
   # Add a test video to data/video/
   python main.py
   ```

2. **Test evaluation**:
   ```bash
   # Ensure ground_truth.json exists
   python evaluate.py
   ```

3. **Test Streamlit app**:
   ```bash
   cd "../Video Streamlit"
   streamlit run app.py
   # Test with both YouTube URL and file upload
   ```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- OpenAI Whisper for audio transcription
- Hugging Face Transformers for NLP models
- Facebook BART for summarization
- ResNet50 for image feature extraction

## Support

For issues and questions:
- Create an issue in the repository
- Check existing issues for solutions
- Review the troubleshooting section

---

**Note**: This project requires significant computational resources. For large-scale processing, consider using a GPU-enabled environment or cloud computing resources.
