"""
Extract key moments from video with timestamps
Based on 2024-2025 research on query-dependent and timestamp-aware summarization
"""

import logging
from typing import List, Dict, Tuple
from transformers import pipeline
import numpy as np

logger = logging.getLogger(__name__)

# Global model cache
_summarization_pipeline = None

def get_summarization_pipeline():
    """Get cached summarization pipeline"""
    global _summarization_pipeline
    if _summarization_pipeline is None:
        logger.info("Loading summarization pipeline for key moment extraction...")
        _summarization_pipeline = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            clean_up_tokenization_spaces=False
        )
    return _summarization_pipeline

def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def extract_key_moments(
    segments: List[Dict],
    num_moments: int = 5,
    min_segment_length: int = 50,
    importance_threshold: float = 0.5
) -> List[Dict]:
    """
    Extract key moments from transcription segments with timestamps.

    Args:
        segments: List of transcription segments from Whisper
                 Each segment has: {'text', 'start', 'end', 'id'}
        num_moments: Number of key moments to extract
        min_segment_length: Minimum characters for a segment to be considered
        importance_threshold: Threshold for importance scoring (0-1)

    Returns:
        List of key moments with timestamps and summaries
    """
    if not segments:
        logger.warning("No segments provided for key moment extraction")
        return []

    logger.info(f"Extracting {num_moments} key moments from {len(segments)} segments")

    # Filter segments by length
    valid_segments = [
        seg for seg in segments
        if len(seg.get('text', '').strip()) >= min_segment_length
    ]

    if not valid_segments:
        logger.warning("No valid segments after filtering")
        return []

    # Score segments based on various features
    scored_segments = []

    for segment in valid_segments:
        text = segment.get('text', '').strip()

        # Calculate importance score
        score = calculate_importance_score(text)

        if score >= importance_threshold:
            scored_segments.append({
                'segment': segment,
                'score': score,
                'text': text
            })

    # Sort by score
    scored_segments.sort(key=lambda x: x['score'], reverse=True)

    # Take top N segments
    top_segments = scored_segments[:num_moments]

    # Sort by timestamp for chronological order
    top_segments.sort(key=lambda x: x['segment'].get('start', 0))

    # Generate summaries for key moments
    key_moments = []
    summarizer = get_summarization_pipeline()

    for idx, item in enumerate(top_segments, 1):
        segment = item['segment']
        text = item['text']

        try:
            # Generate a brief summary
            if len(text) > 100:
                summary_result = summarizer(
                    text,
                    max_length=50,
                    min_length=10,
                    do_sample=False
                )
                summary = summary_result[0]['summary_text']
            else:
                summary = text

            key_moment = {
                'id': idx,
                'timestamp': format_timestamp(segment.get('start', 0)),
                'timestamp_seconds': segment.get('start', 0),
                'end_timestamp': format_timestamp(segment.get('end', 0)),
                'end_timestamp_seconds': segment.get('end', 0),
                'duration': segment.get('end', 0) - segment.get('start', 0),
                'text': text,
                'summary': summary,
                'importance_score': round(item['score'], 3)
            }

            key_moments.append(key_moment)
            logger.debug(f"Key moment {idx}: {key_moment['timestamp']} - {summary[:50]}")

        except Exception as e:
            logger.warning(f"Failed to summarize segment {idx}: {e}")
            continue

    logger.info(f"Extracted {len(key_moments)} key moments")
    return key_moments

def calculate_importance_score(text: str) -> float:
    """
    Calculate importance score for a text segment.

    Uses multiple heuristics:
    - Length (longer segments often contain more information)
    - Question presence (questions often mark important points)
    - Keyword presence (certain words indicate importance)
    - Sentence structure

    Args:
        text: Text segment to score

    Returns:
        Importance score (0-1)
    """
    score = 0.0
    text_lower = text.lower()

    # Length score (normalized)
    length_score = min(len(text) / 500, 1.0) * 0.3
    score += length_score

    # Question indicator (questions often mark key points)
    if '?' in text:
        score += 0.2

    # Important keywords
    important_keywords = [
        'important', 'key', 'critical', 'essential', 'significant',
        'main', 'primary', 'fundamental', 'conclusion', 'summary',
        'first', 'second', 'third', 'finally', 'in conclusion',
        'therefore', 'thus', 'consequently', 'as a result',
        'however', 'but', 'although', 'despite', 'nevertheless'
    ]

    keyword_count = sum(1 for keyword in important_keywords if keyword in text_lower)
    keyword_score = min(keyword_count / 5, 1.0) * 0.3
    score += keyword_score

    # Sentence completeness (segments with complete sentences are more valuable)
    sentences = text.split('.')
    complete_sentences = sum(1 for s in sentences if len(s.strip()) > 10)
    sentence_score = min(complete_sentences / 3, 1.0) * 0.2
    score += sentence_score

    # Normalize to 0-1 range
    return min(score, 1.0)

def generate_chapter_markers(
    key_moments: List[Dict],
    max_chapters: int = 10
) -> List[Dict]:
    """
    Generate chapter markers from key moments.

    Args:
        key_moments: List of key moments
        max_chapters: Maximum number of chapters

    Returns:
        List of chapter markers
    """
    if not key_moments:
        return []

    chapters = []

    for idx, moment in enumerate(key_moments[:max_chapters]):
        chapter = {
            'chapter_number': idx + 1,
            'title': moment['summary'],
            'timestamp': moment['timestamp'],
            'timestamp_seconds': moment['timestamp_seconds'],
            'description': moment['text'][:200] + '...' if len(moment['text']) > 200 else moment['text']
        }
        chapters.append(chapter)

    return chapters

def export_key_moments(
    key_moments: List[Dict],
    output_path: str,
    format: str = 'txt'
) -> str:
    """
    Export key moments to file.

    Args:
        key_moments: List of key moments
        output_path: Path to save file
        format: Export format (txt, json, srt)

    Returns:
        Path to exported file
    """
    import json

    if format == 'txt':
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("KEY MOMENTS\n")
            f.write("=" * 80 + "\n\n")

            for moment in key_moments:
                f.write(f"[{moment['timestamp']}] - {moment['summary']}\n")
                f.write(f"{moment['text']}\n")
                f.write(f"Importance: {moment['importance_score']}\n")
                f.write("-" * 80 + "\n\n")

    elif format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(key_moments, f, indent=2, ensure_ascii=False)

    elif format == 'srt':
        # SRT subtitle format for video players
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, moment in enumerate(key_moments, 1):
                f.write(f"{idx}\n")
                start = format_srt_timestamp(moment['timestamp_seconds'])
                end = format_srt_timestamp(moment['end_timestamp_seconds'])
                f.write(f"{start} --> {end}\n")
                f.write(f"{moment['summary']}\n\n")

    logger.info(f"Exported key moments to {output_path}")
    return output_path

def format_srt_timestamp(seconds: float) -> str:
    """Format timestamp for SRT subtitle format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

if __name__ == "__main__":
    # Test with sample segments
    sample_segments = [
        {
            'text': 'This is an important introduction to the topic. We will cover key concepts.',
            'start': 0.0,
            'end': 5.0,
            'id': 0
        },
        {
            'text': 'However, there are some critical points to consider. First, we need to understand the fundamental principles.',
            'start': 10.0,
            'end': 18.0,
            'id': 1
        },
        {
            'text': 'In conclusion, these findings demonstrate significant implications for the field.',
            'start': 120.0,
            'end': 127.0,
            'id': 2
        }
    ]

    moments = extract_key_moments(sample_segments, num_moments=3)

    print("Key Moments Extracted:")
    print("=" * 80)
    for moment in moments:
        print(f"[{moment['timestamp']}] {moment['summary']}")
        print(f"Score: {moment['importance_score']}")
        print()
