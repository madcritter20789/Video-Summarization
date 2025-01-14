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
import faiss
import logging
from transformers import pipeline

def clean_transcript(transcript):
    # Split into sentences and clean text
    sentences = transcript.split("\n")
    cleaned_sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip() and len(sentence.split()) > 3  # Remove short or empty lines
    ]
    return " ".join(cleaned_sentences)

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


'''
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
from transformers import pipeline
'''


def generate_text_summary(transcript_path):
    # Load summarization pipeline
    summarizer = pipeline("summarization", model="t5-small")

    # Read and clean the transcript
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()
    cleaned_transcript = clean_transcript(transcript)

    # Ensure the transcript is not empty
    if not cleaned_transcript.strip():
        raise ValueError("Transcript is empty or too short after cleaning.")

    # Generate summary
    summary = summarizer(cleaned_transcript, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']


def combine_insights(video_summary, transcript_summary):
    return f"The video showcases {video_summary}. Key highlights include: {transcript_summary}."
