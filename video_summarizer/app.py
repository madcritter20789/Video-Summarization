
'''
import streamlit as st
import os
import shutil
from pytube import YouTube
from main import process_video  # Import the function to process videos
import yt_dlp

# Define paths
UPLOAD_DIR = "data/video/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Function to download a YouTube video
def download_youtube_video(youtube_url):
    """Download a YouTube video using yt-dlp."""
    try:
        ydl_opts = {
            "format": "bestvideo+bestaudio/best",
            "outtmpl": "data/video/%(title)s.%(ext)s",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            video_filename = ydl.prepare_filename(info_dict)
        return video_filename
    except Exception as e:
        st.error(f"Error downloading video: {e}")
        return None


# Streamlit UI
st.title("🎬 Video Summarizer")
st.write("Upload a video or provide a YouTube link to generate a summary.")

# Option to upload video
uploaded_video = st.file_uploader("Upload a video file (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

# Option to enter YouTube link
youtube_url = st.text_input("Or enter a YouTube link:")

# Process button
if st.button("Generate Summary"):
    video_path = None

    # Process uploaded video
    if uploaded_video is not None:
        video_path = os.path.join(UPLOAD_DIR, uploaded_video.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
        st.success(f"Uploaded video: {uploaded_video.name}")

    # Process YouTube video
    elif youtube_url:
        st.info("Downloading video from YouTube...")
        video_path = download_youtube_video(youtube_url)
        if video_path:
            st.success(f"Downloaded YouTube video: {os.path.basename(video_path)}")

    # If no video is provided
    if not video_path:
        st.warning("Please upload a video or enter a YouTube link.")
    else:
        # Run the video processing function
        st.info("Processing video, please wait...")
        summary_path = process_video(video_path)

        # Display summary
        if summary_path:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_text = f.read()
            st.subheader("📄 Generated Summary:")
            st.write(summary_text)
        else:
            st.error("Error generating summary.")
'''

import streamlit as st
import os
import re
from main import process_video  # Import the processing function
from pytube import YouTube
import shutil

# Ensure required folders exist
os.makedirs("data/video", exist_ok=True)
os.makedirs("data/frames", exist_ok=True)

# Function to clean filenames
def clean_filename(filename):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', filename)  # Replace special characters

st.title("📹 Video Summarizer")
st.write("Upload a video file or provide a YouTube link to generate a summary.")

# File Upload
uploaded_file = st.file_uploader("Upload a video file (MP4, AVI, MOV)", type=["mp4", "avi", "mov", "mpeg4"])

# YouTube Link Input
youtube_link = st.text_input("Or enter a YouTube link:")

# Button to start processing
if st.button("Generate Summary"):
    if uploaded_file is not None:
        # Save uploaded file
        file_path = os.path.join("data/video", clean_filename(uploaded_file.name))
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        video_path = file_path  # Set video path for processing
        st.success(f"Uploaded file: {uploaded_file.name}")

    elif youtube_link:
        try:
            yt = YouTube(youtube_link)
            stream = yt.streams.get_highest_resolution()
            clean_title = clean_filename(yt.title) + ".mp4"  # Sanitize filename
            video_path = os.path.join("data/video", clean_title)
            stream.download(filename=video_path)  # Download video
            st.success(f"Downloaded YouTube video: {yt.title}")
        except Exception as e:
            st.error(f"Error downloading video: {str(e)}")
            st.stop()  # Stop execution

    else:
        st.warning("Please upload a video or enter a YouTube link.")
        st.stop()  # Stop execution

    # Process Video
    st.info("Processing video, please wait...")
    summary_path = process_video(video_path)

    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = f.read()
        st.success("✅ Summary Generated:")
        st.text_area("Summary", summary, height=200)
    else:
        st.error("❌ Error processing video. Check logs.")

