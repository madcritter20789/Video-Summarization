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
'''
def summarize(video_index, transcript_index):
    import faiss
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
