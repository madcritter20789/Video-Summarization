'''import os
from src.extract_frames import extract_frames
from src.extract_audio import extract_audio
from src.transcribe_audio import transcribe_audio
from src.vectorize_video import vectorize_frames
from src.vectorize_transcript import vectorize_transcript
from src.summarize import summarize

def main(video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_folder = f"data/frames/{base_name}/"
    audio_path = f"data/audio/{base_name}.mp3"
    transcript_path = f"data/transcripts/{base_name}.txt"
    video_vector_path = f"embeddings/video_vectors/{base_name}.index"
    transcript_vector_path = f"embeddings/transcript_vectors/{base_name}.index"

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

    # Step 6: Summarize
    summary = summarize(video_vector_path, transcript_vector_path)
    print("Summary:", summary)

# Run the process
# main('data/video/sample.mp4')'''
from src.extract_frames import extract_frames
from src.extract_audio import extract_audio
from src.transcribe_audio import transcribe_audio
from src.vectorize_video import vectorize_frames
from src.vectorize_transcript import vectorize_transcript
from src.summarize import summarize
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

    print("Step 1: Extracting frames...")
    extract_frames(video_path, frame_folder)

    print("Step 2: Extracting audio...")
    extract_audio(video_path, audio_path)

    print("Step 3: Transcribing audio...")
    transcribe_audio(audio_path, transcript_path)

    print("Step 4: Vectorizing video frames...")
    vectorize_frames(frame_folder, video_vector_path)

    print("Step 5: Vectorizing transcript...")
    vectorize_transcript(transcript_path, transcript_vector_path)

    print("Step 6: Summarizing video...")
    summary = summarize(video_vector_path, transcript_vector_path)
    print("Summary:", summary)

# Replace 'sample.mp4' with the name of your video file
main('data/video/sample.mp4')
