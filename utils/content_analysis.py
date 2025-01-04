"""
from transformers import pipeline

def analyze_and_summarize(content):

    'Summarizes and analyzes the given transcript.'

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(content, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]

    # Generate basic insights
    word_count = len(content.split())
    keywords = [word for word in set(content.split()) if len(word) > 6][:10]

    insights = {
        "word_count": word_count,
        "keywords": keywords,
    }

    return summary, insights

from transformers import pipeline


def analyze_and_summarize(content):

    'Summarizes and analyzes the given transcript.'

    if not content or len(content.strip()) == 0:
        raise ValueError("Content is empty or invalid.")
    if len(content) > 50000:  # Arbitrary limit to prevent extreme cases
        raise ValueError("Content is too long to process in one go.")

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Break content into chunks (BART max token limit is 1024)
    max_token_length = 1024
    chunks = [content[i:i + max_token_length] for i in range(0, len(content), max_token_length)]

    # Summarize each chunk and combine results
    summaries = []
    for chunk in chunks:
        summaries.append(summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]["summary_text"])

    summary = " ".join(summaries)

    # Generate basic insights
    word_count = len(content.split())
    keywords = [word for word in set(content.split()) if len(word) > 6][:10]

    insights = {
        "word_count": word_count,
        "keywords": keywords,
    }

    return summary, insights
    """
from transformers import pipeline


def analyze_and_summarize(content):
    """
    Summarizes and analyzes the given transcript with error handling.
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Define max token length for BART model
    max_token_length = 1024
    chunks = [content[i:i + max_token_length] for i in range(0, len(content), max_token_length)]

    summaries = []
    errors = []

    for idx, chunk in enumerate(chunks):
        try:
            # Summarize each chunk
            summary_chunk = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
            summaries.append(summary_chunk)
        except Exception as e:
            error_message = f"Error summarizing chunk {idx + 1}: {str(e)}"
            print(error_message)  # Log to console (or use a logging library)
            errors.append(error_message)

    # Combine successful summaries
    summary = " ".join(summaries)

    # Generate basic insights even if summarization partially failed
    word_count = len(content.split())
    keywords = [word for word in set(content.split()) if len(word) > 6][:10]

    insights = {
        "word_count": word_count,
        "keywords": keywords,
        "errors": errors,  # Include errors in insights for debugging
    }

    # If no summaries were successful, raise an exception
    if not summaries:
        raise ValueError("All chunks failed to summarize. See 'errors' for details.")

    return summary, insights

