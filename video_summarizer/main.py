
import os
from src.extract_frames import extract_frames
from src.extract_audio import extract_audio
from src.transcribe_audio import transcribe_audio
from src.vectorize_video import vectorize_frames
from src.vectorize_transcript import vectorize_transcript
from src.summarize import generate_detailed_summary
from src.combine_embeddings import combine_embeddings
from src.train_model import train_model

# Directory paths
VIDEO_DIR = "data/video/"
FRAME_DIR = "data/frames/"
AUDIO_DIR = "data/audio/"
TRANSCRIPT_DIR = "data/transcripts/"
VECTOR_DIR = "embeddings/"
SUMMARY_DIR = "results/summaries/"
MODEL_DIR = "models/"


def process_video(video_path):
    """Processes a single video: extracts frames, audio, transcript, embeddings, and summary."""

    base_name = os.path.splitext(os.path.basename(video_path))[0]  # Get video filename (without extension)

    # Define paths for extracted data
    frame_folder = os.path.join(FRAME_DIR, base_name)
    audio_path = os.path.join(AUDIO_DIR, f"{base_name}.mp3")
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{base_name}.txt")
    video_vector_path = os.path.join(VECTOR_DIR, "video_vectors", f"{base_name}.index")
    transcript_vector_path = os.path.join(VECTOR_DIR, "transcript_vectors", f"{base_name}.index")
    combined_vector_path = os.path.join(VECTOR_DIR, "combined_vectors", f"{base_name}.index")
    summary_path = os.path.join(SUMMARY_DIR, f"{base_name}_summary.txt")

    # Step 1: Extract frames
    extract_frames(video_path, frame_folder)

    # Step 2: Extract audio
    extract_audio(video_path, audio_path)

    # Step 3: Transcribe audio
    transcribe_audio(audio_path, transcript_path)

    # Step 4: Vectorize video frames
    vectorize_frames(frame_folder, video_vector_path)

    # Step 5: Vectorize transcript
    vectorize_transcript(transcript_path, transcript_vector_path)

    # Step 6: Combine embeddings
    combine_embeddings(video_vector_path, transcript_vector_path, combined_vector_path)

    # Step 7: Generate summary
    summary = generate_detailed_summary(transcript_path)

    # Save summary
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Summary saved to {summary_path}")

    return combined_vector_path


def main():
    """Processes all videos in the `data/video/` directory."""

    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith((".mp4", ".avi", ".mov"))]

    if not video_files:
        print("No videos found in data/video/ directory.")
        return

    combined_vectors = []

    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        print(f"\n🔹 Processing video: {video_file}")
        combined_vector_path = process_video(video_path)
        combined_vectors.append(combined_vector_path)

    # Optional: Train models after processing all videos
    lstm_model_path = os.path.join(MODEL_DIR, "lstm_model.pth")
    transformer_model_path = os.path.join(MODEL_DIR, "transformer_model")
    summaries_path = os.path.join(SUMMARY_DIR, "ground_truth.json")

    print("\n🚀 Training models on combined data...")
    train_model(combined_vectors, summaries_path, lstm_model_path, transformer_model_path)


if __name__ == "__main__":
    main()

'''


import os
from src.extract_frames import extract_frames
from src.extract_audio import extract_audio
from src.transcribe_audio import transcribe_audio
from src.vectorize_video import vectorize_frames
from src.vectorize_transcript import vectorize_transcript
from src.summarize import generate_detailed_summary
from src.combine_embeddings import combine_embeddings
from src.train_model import train_model

# Directory paths
FRAME_DIR = "data/frames/"
AUDIO_DIR = "data/audio/"
TRANSCRIPT_DIR = "data/transcripts/"
VECTOR_DIR = "embeddings/"
SUMMARY_DIR = "results/summaries/"
MODEL_DIR = "models/"

def process_video(video_path):
    """Processes only the given video and ignores others in the folder."""

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    base_name = os.path.splitext(os.path.basename(video_path))[0]  # Extract filename

    # Define paths for extracted data
    frame_folder = os.path.join(FRAME_DIR, base_name)
    audio_path = os.path.join(AUDIO_DIR, f"{base_name}.mp3")
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{base_name}.txt")
    video_vector_path = os.path.join(VECTOR_DIR, "video_vectors", f"{base_name}.index")
    transcript_vector_path = os.path.join(VECTOR_DIR, "transcript_vectors", f"{base_name}.index")
    combined_vector_path = os.path.join(VECTOR_DIR, "combined_vectors", f"{base_name}.index")
    summary_path = os.path.join(SUMMARY_DIR, f"{base_name}_summary.txt")

    # Step 1: Extract frames
    extract_frames(video_path, frame_folder)

    # Step 2: Extract audio
    extract_audio(video_path, audio_path)

    # Step 3: Transcribe audio
    transcribe_audio(audio_path, transcript_path)

    # Step 4: Vectorize video frames
    vectorize_frames(frame_folder, video_vector_path)

    # Step 5: Vectorize transcript
    vectorize_transcript(transcript_path, transcript_vector_path)

    # Step 6: Combine embeddings
    combine_embeddings(video_vector_path, transcript_vector_path, combined_vector_path)

    # Step 7: Generate summary
    summary = generate_detailed_summary(transcript_path)

    # Save summary
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Summary saved to {summary_path}")

    return summary_path  # Return the summary file path
'''