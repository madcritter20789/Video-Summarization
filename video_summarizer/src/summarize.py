'''from langchain.vectorstores import faiss

def summarize(video_index, transcript_index):
    # Load FAISS indices
    video_idx = faiss.read_index(video_index)
    transcript_idx = faiss.read_index(transcript_index)

    # Simple heuristic: Combine top-k nearest embeddings from both
    k = 5
    video_vectors = video_idx.reconstruct_n(0, k)
    transcript_vectors = transcript_idx.reconstruct_n(0, k)

    # Example: Use nearest transcript vectors to label video frames
    insights = {
        "video_summary": video_vectors.tolist(),
        "transcript_summary": transcript_vectors.tolist()
    }
    return insights

# Example usage
# summary = summarize('embeddings/video_vectors/sample.index', 'embeddings/transcript_vectors/sample.index')



def summarize(video_index, transcript_index):

    print("Loading video index...")
    video_idx = faiss.read_index(video_index)
    print("Loading transcript index...")
    transcript_idx = faiss.read_index(transcript_index)

    print("Querying top vectors...")
    k = 5
    video_vectors = video_idx.reconstruct_n(0, k)
    transcript_vectors = transcript_idx.reconstruct_n(0, k)

    insights = {
        "video_summary": video_vectors.tolist(),
        "transcript_summary": transcript_vectors.tolist()
    }
    print("Generated insights:", insights)
    return insights


'''
'''
import faiss
import logging
from transformers import pipeline
import re

def clean_transcript(transcript):
    # Remove non-verbal sounds or annotations (e.g., [laughs], [music])
    transcript = re.sub(r"\[.*?\]", "", transcript)
    # Fix sentence boundaries and punctuation
    transcript = re.sub(r"([a-z])([A-Z])", r"\1. \2", transcript)
    # Remove excessive whitespace
    transcript = re.sub(r"\s+", " ", transcript).strip()
    return transcript


def summarize(video_vector_path, transcript_vector_path):
    print("Loading video index...")
    video_idx = faiss.read_index(video_vector_path)
    print("Loading transcript index...")
    transcript_idx = faiss.read_index(transcript_vector_path)

    print(f"Transcript Index Size: {transcript_idx.ntotal}")
    print(f"Video Index Size: {video_idx.ntotal}")
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Video Index Size: {video_idx.ntotal}")
    logging.info(f"Transcript Index Size: {transcript_idx.ntotal}")


    if video_idx.ntotal == 0 or transcript_idx.ntotal == 0:
        raise ValueError("One or both FAISS indices are empty. Check vectorization steps.")
    print("Querying top vectors...")
    k = 10  # Number of top vectors to query
    k = min(k, transcript_idx.ntotal)  # Ensure k does not exceed available vectors
    transcript_vectors = transcript_idx.reconstruct_n(0, k)

    if transcript_idx.ntotal == 0:
        raise RuntimeError("Transcript index is empty. Ensure the transcript has sufficient content.")
    elif transcript_idx.ntotal < k:
        print(f"Warning: Requested {k} vectors, but only {transcript_idx.ntotal} are available.")
        k = transcript_idx.ntotal

    try:
        transcript_vectors = transcript_idx.reconstruct_n(0, k)
        video_vectors = video_idx.reconstruct_n(0, k)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to reconstruct vectors: {e}")
    video_vectors = video_idx.reconstruct_n(0, k)
    transcript_vectors = transcript_idx.reconstruct_n(0, k)

    insights = {
        "video_summary": video_vectors.tolist(),
        "transcript_summary": transcript_vectors.tolist()
    }
    print("Generated insights:", insights)
    return insights
    # Continue with summarization logic...

def generate_text_summary(transcript_path):
    # Load summarization pipeline
    from transformers import pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # Read and clean the transcript
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()
    cleaned_transcript = clean_transcript(transcript)

    # Ensure transcript is not too long for the model
    max_chunk_size = 1024
    chunks = [cleaned_transcript[i:i + max_chunk_size] for i in range(0, len(cleaned_transcript), max_chunk_size)]
    summaries = [summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text'] for chunk in
                 chunks]

    return " ".join(summaries)


def combine_insights(video_summary, transcript_summary):
    return (
        f"The video showcases visually stunning moments, including {video_summary}. "
        f"Key highlights from the narrative include: {transcript_summary}."
    )

def generate_text_summary(transcript_path):
    # Load the summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Read the transcript
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    # Ensure the transcript is not empty
    if not transcript.strip():
        raise ValueError("Transcript is empty. Ensure the audio transcription step was successful.")

    # Generate summary
    summary = summarizer(transcript, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']
    
    ##################################################
from transformers import pipeline



def generate_text_summary(transcript_path):

    summarizer = pipeline("summarization", model="t5-small")


    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()
    cleaned_transcript = clean_transcript(transcript)

    # Ensure the transcript is not empty
    if not cleaned_transcript.strip():
        raise ValueError("Transcript is empty or too short after cleaning.")

    # Generate summary
    summary = summarizer(cleaned_transcript, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']
'''

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
