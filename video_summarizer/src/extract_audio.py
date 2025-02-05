
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
