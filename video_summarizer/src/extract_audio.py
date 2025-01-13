'''
import os
from moviepy import VideoFileClip


def extract_audio(video_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_path)

# Example usage
# extract_audio('data/video/sample.mp4', 'data/audio/sample.mp3')
'''
import subprocess

def extract_audio(video_path, audio_path):
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-q:a", "0",
        "-map", "a",
        audio_path
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"Audio extracted to {audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
