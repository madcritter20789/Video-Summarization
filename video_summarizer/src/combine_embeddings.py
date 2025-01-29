import numpy as np
import faiss

def combine_embeddings(video_vector_path, transcript_vector_path, combined_vector_path):
    # Load video and transcript vectors
    video_vectors = faiss.read_index(video_vector_path).reconstruct_n(0, faiss.read_index(video_vector_path).ntotal)
    transcript_vectors = faiss.read_index(transcript_vector_path).reconstruct_n(0, faiss.read_index(transcript_vector_path).ntotal)

    # Align dimensions
    if transcript_vectors.shape[0] == 1:
        transcript_vectors = np.tile(transcript_vectors, (video_vectors.shape[0], 1))
    elif video_vectors.shape[0] != transcript_vectors.shape[0]:
        min_length = min(video_vectors.shape[0], transcript_vectors.shape[0])
        video_vectors = video_vectors[:min_length]
        transcript_vectors = transcript_vectors[:min_length]

    # Combine vectors
    combined_vectors = np.hstack((video_vectors, transcript_vectors))

    # Save combined vectors
    index = faiss.IndexFlatL2(combined_vectors.shape[1])
    index.add(combined_vectors)
    faiss.write_index(index, combined_vector_path)
    print(f"Combined vectors saved to {combined_vector_path}")
