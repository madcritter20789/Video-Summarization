"""
Enhanced Video Processing Pipeline with Modern Features
- Progress tracking
- Caching
- Structured logging
- Key moments extraction
- Configuration management
"""

import os
import sys
import json
import logging
from pathlib import Path
from tqdm import tqdm

# Import configuration
from config import (
    setup_logging, setup_directories,
    VIDEO_CONFIG, TRANSCRIPTION_CONFIG, SUMMARIZATION_CONFIG
)

# Import processing modules
from src.extract_frames import extract_frames
from src.extract_audio import extract_audio
from src.transcribe_audio import transcribe_audio
from src.vectorize_video import vectorize_frames
from src.vectorize_transcript import vectorize_transcript
from src.summarize import generate_detailed_summary
from src.combine_embeddings import combine_embeddings
from src.extract_key_moments import extract_key_moments, export_key_moments

# Setup logging
logger = setup_logging()

# Directory paths
VIDEO_DIR = "data/video/"
FRAME_DIR = "data/frames/"
AUDIO_DIR = "data/audio/"
TRANSCRIPT_DIR = "data/transcripts/"
VECTOR_DIR = "embeddings/"
SUMMARY_DIR = "results/summaries/"

def process_video_enhanced(video_path: str, extract_moments: bool = True) -> dict:
    """
    Process a single video with all modern features.

    Args:
        video_path: Path to video file
        extract_moments: Whether to extract key moments with timestamps

    Returns:
        Dictionary with all processing results
    """
    logger.info(f"=" * 80)
    logger.info(f"Processing video: {video_path}")
    logger.info(f"=" * 80)

    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    results = {
        'video_name': base_name,
        'video_path': video_path,
        'steps_completed': []
    }

    # Define paths for extracted data
    frame_folder = os.path.join(FRAME_DIR, base_name)
    audio_path = os.path.join(AUDIO_DIR, f"{base_name}.mp3")
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{base_name}.txt")
    transcript_json_path = os.path.join(TRANSCRIPT_DIR, f"{base_name}_full.json")
    video_vector_path = os.path.join(VECTOR_DIR, "video_vectors", f"{base_name}.index")
    transcript_vector_path = os.path.join(VECTOR_DIR, "transcript_vectors", f"{base_name}.index")
    combined_vector_path = os.path.join(VECTOR_DIR, "combined_vectors", f"{base_name}.index")
    summary_path = os.path.join(SUMMARY_DIR, f"{base_name}_summary.txt")
    key_moments_path = os.path.join(SUMMARY_DIR, f"{base_name}_key_moments.json")

    try:
        # Step 1: Extract frames
        logger.info("Step 1/8: Extracting frames...")
        num_frames = extract_frames(
            video_path,
            frame_folder,
            frame_rate=VIDEO_CONFIG['frame_rate']
        )
        results['steps_completed'].append('extract_frames')
        results['num_frames'] = num_frames
        logger.info(f"✓ Extracted {num_frames} frames")

        # Step 2: Extract audio
        logger.info("Step 2/8: Extracting audio...")
        extract_audio(video_path, audio_path)
        results['steps_completed'].append('extract_audio')
        results['audio_path'] = audio_path
        logger.info(f"✓ Audio extracted to {audio_path}")

        # Step 3: Transcribe audio
        logger.info("Step 3/8: Transcribing audio...")
        transcription_result = transcribe_audio(
            audio_path,
            transcript_path,
            model_name=TRANSCRIPTION_CONFIG['model_size'],
            language=TRANSCRIPTION_CONFIG['language']
        )
        results['steps_completed'].append('transcribe_audio')
        results['transcript_path'] = transcript_path
        results['language'] = transcription_result.get('language', 'unknown')

        # Save full transcription with segments
        with open(transcript_json_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_result, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Transcription completed (Language: {results['language']})")

        # Step 4: Extract key moments (if enabled)
        if extract_moments and transcription_result.get('segments'):
            logger.info("Step 4/8: Extracting key moments...")
            key_moments = extract_key_moments(
                segments=transcription_result['segments'],
                num_moments=SUMMARIZATION_CONFIG['num_key_moments'],
                importance_threshold=0.5
            )
            results['key_moments'] = key_moments

            # Export key moments
            export_key_moments(key_moments, key_moments_path, format='json')
            export_key_moments(
                key_moments,
                key_moments_path.replace('.json', '.txt'),
                format='txt'
            )
            export_key_moments(
                key_moments,
                key_moments_path.replace('.json', '.srt'),
                format='srt'
            )

            results['steps_completed'].append('extract_key_moments')
            logger.info(f"✓ Extracted {len(key_moments)} key moments")
        else:
            logger.info("Step 4/8: Skipping key moments extraction")

        # Step 5: Vectorize video frames
        logger.info("Step 5/8: Vectorizing video frames...")
        vectorize_frames(frame_folder, video_vector_path)
        results['steps_completed'].append('vectorize_frames')
        results['video_vector_path'] = video_vector_path
        logger.info(f"✓ Video frames vectorized")

        # Step 6: Vectorize transcript
        logger.info("Step 6/8: Vectorizing transcript...")
        vectorize_transcript(transcript_path, transcript_vector_path)
        results['steps_completed'].append('vectorize_transcript')
        results['transcript_vector_path'] = transcript_vector_path
        logger.info(f"✓ Transcript vectorized")

        # Step 7: Combine embeddings
        logger.info("Step 7/8: Combining embeddings...")
        combine_embeddings(video_vector_path, transcript_vector_path, combined_vector_path)
        results['steps_completed'].append('combine_embeddings')
        results['combined_vector_path'] = combined_vector_path
        logger.info(f"✓ Embeddings combined")

        # Step 8: Generate summary
        logger.info("Step 8/8: Generating summary...")
        summary = generate_detailed_summary(transcript_path)

        # Save summary
        os.makedirs(SUMMARY_DIR, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

        results['steps_completed'].append('generate_summary')
        results['summary_path'] = summary_path
        results['summary'] = summary
        logger.info(f"✓ Summary generated and saved to {summary_path}")

        # Success
        results['status'] = 'success'
        results['message'] = 'Video processed successfully'

        logger.info(f"=" * 80)
        logger.info(f"✓ Processing completed successfully!")
        logger.info(f"=" * 80)

        return results

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        results['status'] = 'error'
        results['error'] = str(e)
        raise

def main():
    """Process all videos in the video directory."""
    logger.info("Video Summarization Pipeline - Enhanced Version")
    logger.info("=" * 80)

    # Setup directories
    setup_directories()

    # Get all video files
    video_files = []
    for ext in VIDEO_CONFIG['supported_formats']:
        video_files.extend(Path(VIDEO_DIR).glob(f"*{ext}"))

    if not video_files:
        logger.warning(f"No videos found in {VIDEO_DIR}")
        logger.info("Please add videos to the data/video/ directory")
        return

    logger.info(f"Found {len(video_files)} video(s) to process")

    # Process each video
    results_summary = []

    for video_file in tqdm(video_files, desc="Processing videos", unit="video"):
        try:
            video_path = str(video_file)
            logger.info(f"\nProcessing: {video_file.name}")

            result = process_video_enhanced(
                video_path,
                extract_moments=SUMMARIZATION_CONFIG['extract_key_moments']
            )

            results_summary.append({
                'video': video_file.name,
                'status': result['status'],
                'steps_completed': len(result['steps_completed']),
                'key_moments': len(result.get('key_moments', [])),
                'summary_path': result.get('summary_path')
            })

        except Exception as e:
            logger.error(f"Failed to process {video_file.name}: {e}")
            results_summary.append({
                'video': video_file.name,
                'status': 'failed',
                'error': str(e)
            })

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 80)

    for result in results_summary:
        status_symbol = "✓" if result['status'] == 'success' else "✗"
        logger.info(f"{status_symbol} {result['video']}: {result['status']}")

        if result['status'] == 'success':
            logger.info(f"  - Steps completed: {result['steps_completed']}/8")
            logger.info(f"  - Key moments: {result.get('key_moments', 0)}")
            logger.info(f"  - Summary: {result.get('summary_path', 'N/A')}")

    successful = sum(1 for r in results_summary if r['status'] == 'success')
    logger.info(f"\nTotal: {successful}/{len(results_summary)} videos processed successfully")

if __name__ == "__main__":
    main()
