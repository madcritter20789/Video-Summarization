import streamlit as st
from main_tool import process_video_or_link
import os

st.title("Video Content Analyzer")
st.write("Upload a video file or provide a YouTube link to analyze its content.")

input_type = st.radio("Select input type:", ["YouTube Link", "Upload Video"])

if input_type == "YouTube Link":
    youtube_link = st.text_input("Enter YouTube Link")
    if youtube_link and st.button("Analyze"):
        with st.spinner("Processing..."):
            try:
                result = process_video_or_link(youtube_link)
                st.success("Analysis Complete!")
                st.subheader("Transcript")
                st.write(result["transcript"])
                st.subheader("Summary")
                st.write(result["summary"])
                st.subheader("Insights")
                st.write(result["insights"])
            except Exception as e:
                st.error(f"Error: {e}")

elif input_type == "Upload Video":
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    if uploaded_file and st.button("Analyze"):
        with st.spinner("Processing..."):
            try:
                temp_path = os.path.join("static/uploaded_videos", uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                result = process_video_or_link(temp_path)
                st.success("Analysis Complete!")
                st.subheader("Transcript")
                st.write(result["transcript"])
                st.subheader("Summary")
                st.write(result["summary"])
                st.subheader("Insights")
                st.write(result["insights"])
            except Exception as e:
                st.error(f"Error: {e}")
