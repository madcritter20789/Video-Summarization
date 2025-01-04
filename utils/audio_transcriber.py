"""
import whisper
from moviepy.editor import VideoFileClip
import os
import tempfile


def transcribe_audio(video_path):

    'Extracts audio from a video and generates a transcript using Whisper.'

    model = whisper.load_model("base")
    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    VideoFileClip(video_path).audio.write_audiofile(temp_audio.name)

    transcription = model.transcribe(temp_audio.name)
    os.remove(temp_audio.name)
    return transcription["text"]
"""
import ffmpeg
import tempfile
import os

def extract_audio_ffmpeg(video_path):
    """
    Extracts audio from a video using FFmpeg.
    """
    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    ffmpeg.input(video_path).output(temp_audio, format="mp3").run(quiet=True, overwrite_output=True)
    return temp_audio
