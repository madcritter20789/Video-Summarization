from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

def download_youtube_video(link, save_path):
    """
    Downloads a YouTube video to the specified directory.
    """
    yt = YouTube(link)
    stream = yt.streams.filter(progressive=True, file_extension="mp4").first()
    return stream.download(save_path)

def get_youtube_transcript(video_id):
    """
    Fetches transcript of a YouTube video if available.
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([item["text"] for item in transcript])
    except TranscriptsDisabled:
        return None
