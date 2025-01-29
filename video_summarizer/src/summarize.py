
from transformers import pipeline
import re

def clean_transcript(transcript):
    """
    Cleans the raw transcript by removing annotations, fixing punctuation, and normalizing whitespace.
    """
    # Remove non-verbal annotations (e.g., [music], [laughs])
    transcript = re.sub(r"\[.*?\]", "", transcript)
    # Fix punctuation and sentence boundaries
    transcript = re.sub(r"([a-z])([A-Z])", r"\1. \2", transcript)
    # Remove excessive whitespace
    transcript = re.sub(r"\s+", " ", transcript).strip()
    return transcript

def generate_detailed_summary(transcript_path):
    """
    Generates a detailed summary from the transcript.
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Read and clean the transcript
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()
    cleaned_transcript = clean_transcript(transcript)

    # Split transcript into manageable chunks for summarization
    max_chunk_size = 1024
    chunks = [cleaned_transcript[i:i+max_chunk_size] for i in range(0, len(cleaned_transcript), max_chunk_size)]

    # Summarize each chunk
    summaries = [summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text'] for chunk in chunks]

    # Combine summaries into a detailed outline
    detailed_summary = organize_summary_into_points(" ".join(summaries))
    return detailed_summary

def organize_summary_into_points(summary_text):
    """
    Organizes the summarized text into categories and points dynamically.
    """
    # Split the summary into sentences
    sentences = summary_text.split(". ")
    categories = {}

    # Categorize sentences based on content
    for sentence in sentences:
        if not sentence.strip():
            continue  # Skip empty sentences
        # Use keywords to detect categories dynamically
        category = detect_category(sentence)
        if category not in categories:
            categories[category] = []
        categories[category].append(sentence.strip())

    # Format categories into a detailed point-based summary
    formatted_summary = ""
    for category, points in categories.items():
        formatted_summary += f"\n**{category}**:\n"
        for i, point in enumerate(points, start=1):
            formatted_summary += f"{i}. {point}.\n"

    return formatted_summary.strip()

def detect_category(sentence):
    """
    Dynamically detects the category of a sentence based on keywords and context.
    """
    keywords = {
        "Introduction": ["introduce", "overview", "philosophy", "beginning"],
        "Critiques": ["critique", "problem", "issue", "challenge"],
        "Tools": ["tool", "software", "application", "platform"],
        "Tips": ["tip", "advice", "recommendation", "suggestion"],
        "Future Plans": ["future", "plan", "upcoming", "next"],
        "Insights": ["insight", "knowledge", "lesson", "understanding"]
    }

    for category, words in keywords.items():
        if any(word in sentence.lower() for word in words):
            return category

    return "General"  # Default category if no keyword matches
