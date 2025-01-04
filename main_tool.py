import os
from utils.downloader import download_youtube_video, get_youtube_transcript
from utils.audio_transcriber import extract_audio_ffmpeg  # Updated function name
from utils.content_analysis import analyze_and_summarize

def process_video_or_link(input_source, save_path="static/uploaded_videos"):
    """
    Processes a YouTube link or uploaded video file and returns analysis results.
    """
    os.makedirs(save_path, exist_ok=True)

    transcript = None
    video_path = None

    if input_source.startswith("http"):
        # Handle YouTube link
        video_id = input_source.split("v=")[-1]
        transcript = get_youtube_transcript(video_id)

        if not transcript or not transcript.strip():
            print("No transcript available from YouTube. Downloading video for audio transcription...")
            video_path = download_youtube_video(input_source, save_path)
            transcript = extract_audio_ffmpeg(video_path)
    else:
        # Handle uploaded video file
        video_path = input_source
        transcript = extract_audio_ffmpeg(video_path)

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty. Unable to process.")

    # Analyze and summarize the transcript
    summary, insights = analyze_and_summarize(transcript)

    return {
        "transcript": transcript,
        "summary": summary,
        "insights": insights,
        "video_path": video_path,
    }
