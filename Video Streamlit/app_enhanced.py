"""
Enhanced Streamlit Video Analyzer with Modern UI/UX
Based on 2024-2025 best practices
"""

import streamlit as st
import os
import sys
from pathlib import Path
import tempfile
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from main_tool import process_video_or_link

# Page configuration
st.set_page_config(
    page_title="Video Summarization AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    .download-button {
        background-color: #28a745;
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/video.png", width=100)
    st.title("⚙️ Settings")

    st.markdown("---")

    # Processing options
    st.subheader("Processing Options")

    whisper_model = st.selectbox(
        "Whisper Model Size",
        ["tiny", "base", "small", "medium", "large"],
        index=1,
        help="Larger models are more accurate but slower"
    )

    language = st.selectbox(
        "Language",
        ["Auto-detect", "English", "Spanish", "French", "German", "Chinese", "Japanese"],
        help="Select language for better transcription accuracy"
    )

    st.markdown("---")

    # Display options
    st.subheader("Display Options")

    show_timestamps = st.checkbox("Show Timestamps", value=True)
    show_keywords = st.checkbox("Show Keywords", value=True)
    show_insights = st.checkbox("Show Insights", value=True)

    st.markdown("---")

    # About section
    st.subheader("About")
    st.info("""
    **Video Summarization AI**

    Advanced AI-powered video analysis tool featuring:
    - 🎯 Accurate transcription
    - 📝 Intelligent summarization
    - 🔍 Keyword extraction
    - ⏱️ Timestamp markers
    - 📊 Content insights

    Built with OpenAI Whisper, BART, and Streamlit.
    """)

# Main content
st.markdown('<div class="main-header">🎬 Video Summarization AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a video or provide a YouTube link for AI-powered analysis</div>', unsafe_allow_html=True)

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📹 YouTube Link", "📁 Upload Video"])

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'results' not in st.session_state:
    st.session_state.results = None

with tab1:
    st.markdown("### Enter YouTube URL")
    youtube_link = st.text_input(
        "YouTube Link",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_youtube = st.button(
            "🚀 Analyze YouTube Video",
            use_container_width=True,
            type="primary"
        )

    if analyze_youtube and youtube_link:
        st.session_state.processing = True
        st.session_state.input_source = youtube_link
        st.session_state.input_type = "youtube"

with tab2:
    st.markdown("### Upload Video File")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi", "mkv", "wmv"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        # Show video preview
        st.video(uploaded_file)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_upload = st.button(
            "🚀 Analyze Uploaded Video",
            use_container_width=True,
            type="primary"
        )

    if analyze_upload and uploaded_file:
        st.session_state.processing = True

        # Save uploaded file temporarily
        temp_dir = Path("static/uploaded_videos")
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_path = temp_dir / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.input_source = str(temp_path)
        st.session_state.input_type = "upload"

# Process video
if st.session_state.processing:
    try:
        # Progress indicator
        progress_text = "Processing video... This may take a few minutes."
        progress_bar = st.progress(0, text=progress_text)

        # Simulate progress updates (in real implementation, integrate with actual processing)
        with st.spinner("🎬 Downloading video..."):
            progress_bar.progress(20, text="Downloading video...")

        with st.spinner("🎤 Extracting audio..."):
            progress_bar.progress(40, text="Extracting audio...")

        with st.spinner("📝 Transcribing audio..."):
            progress_bar.progress(60, text="Transcribing audio with Whisper...")

            # Process the video
            result = process_video_or_link(st.session_state.input_source)

        with st.spinner("🤖 Generating summary..."):
            progress_bar.progress(80, text="Generating AI summary...")

        progress_bar.progress(100, text="✅ Processing complete!")

        # Store results
        st.session_state.results = result
        st.session_state.processing = False

        # Clear progress bar
        progress_bar.empty()

        st.success("✅ Analysis Complete!")

    except Exception as e:
        st.session_state.processing = False
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

# Display results
if st.session_state.results:
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")

    result = st.session_state.results

    # Create columns for results
    col1, col2 = st.columns([2, 1])

    with col1:
        # Transcript
        with st.expander("📝 Full Transcript", expanded=False):
            st.text_area(
                "Transcript",
                value=result["transcript"],
                height=300,
                label_visibility="collapsed"
            )

            # Download button for transcript
            st.download_button(
                label="⬇️ Download Transcript",
                data=result["transcript"],
                file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

        # Summary
        st.markdown("### 📄 Summary")
        st.markdown(f"""
        <div class="success-box">
        {result["summary"]}
        </div>
        """, unsafe_allow_html=True)

        # Download button for summary
        st.download_button(
            label="⬇️ Download Summary",
            data=result["summary"],
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

    with col2:
        # Insights
        if show_insights:
            st.markdown("### 💡 Insights")

            insights = result["insights"]

            # Word count
            st.metric("Word Count", insights.get("word_count", 0))

            # Keywords
            if show_keywords and "keywords" in insights:
                st.markdown("**🔑 Keywords:**")
                keywords = insights["keywords"][:10]  # Limit to 10
                for i, keyword in enumerate(keywords, 1):
                    st.markdown(f"{i}. `{keyword}`")

            # Errors (if any)
            if "errors" in insights and insights["errors"]:
                with st.expander("⚠️ Processing Warnings", expanded=False):
                    for error in insights["errors"]:
                        st.warning(error)

    # Download all results as JSON
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        results_json = json.dumps({
            "timestamp": datetime.now().isoformat(),
            "input_type": st.session_state.input_type,
            "transcript": result["transcript"],
            "summary": result["summary"],
            "insights": result["insights"]
        }, indent=2)

        st.download_button(
            label="📦 Download Complete Analysis (JSON)",
            data=results_json,
            file_name=f"video_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    # New analysis button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Analyze Another Video", use_container_width=True):
            st.session_state.results = None
            st.session_state.processing = False
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Built with ❤️ using Streamlit, OpenAI Whisper, and Transformers</p>
    <p style="font-size: 0.8rem;">
        🚀 Powered by AI | 📊 Real-time Processing | 🔒 Privacy-Focused
    </p>
</div>
""", unsafe_allow_html=True)
