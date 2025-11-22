import whisper
import ffmpeg
import tempfile
import os

def extract_audio_ffmpeg(video_path):
    """
    Extracts audio from a video using FFmpeg.
    """
    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    try:
        ffmpeg.input(video_path).output(temp_audio, format="mp3").run(quiet=True, overwrite_output=True)
        return temp_audio
    except ffmpeg.Error as e:
        print(f"Error extracting audio with ffmpeg: {e}")
        raise

def transcribe_audio_from_file(audio_path):
    """
    Transcribes audio from a file using Whisper.
    """
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        raise

