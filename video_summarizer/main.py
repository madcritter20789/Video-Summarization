from src.extract_frames import extract_frames
from src.extract_audio import extract_audio
from src.transcribe_audio import transcribe_audio
from src.vectorize_video import vectorize_frames
from src.vectorize_transcript import vectorize_transcript
from src.summarize import generate_detailed_summary
from src.combine_embeddings import combine_embeddings
from src.train_model import train_model, fine_tune_transformer
import os

def main(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_folder = f"data/frames/{base_name}/"
    audio_path = f"data/audio/{base_name}.mp3"
    transcript_path = f"data/transcripts/{base_name}.txt"
    video_vector_path = f"embeddings/video_vectors/{base_name}.index"
    transcript_vector_path = f"embeddings/transcript_vectors/{base_name}.index"
    combined_vector_path = f"embeddings/combined_vectors/{base_name}.index"
    summary_path = f"results/summaries/{base_name}_summary.txt"

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

    # Step 7: Generate detailed summary
    detailed_summary = generate_detailed_summary(transcript_path)

    # Save the summary
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(detailed_summary)
    print(f"Summary saved to {summary_path}")

    # Optional: Train models if needed
    lstm_model_path = "models/lstm_model.pth"
    transformer_model_path = "models/transformer_model"
    summaries_path = "results/summaries/ground_truth.json"

    train_model(combined_vector_path, summaries_path, lstm_model_path, transformer_model_path)

# Example usage
if __name__ == "__main__":
    video_file = "data/video/sample.mp4"
    main(video_file)
