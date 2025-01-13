from safetensors import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
#import os
import torch
'''
def vectorize_transcript(transcript_path, output_file):


    with open(transcript_path, "r") as f:
        sentences = f.readlines()
'''


# In the vectorize_transcript function
def vectorize_transcript(transcript_path, transcript_vector_path):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    # Open the transcript file with the correct encoding
    with open(transcript_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()

    vectors = []
    for sentence in sentences:
        tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = model(**tokens).last_hidden_state.mean(dim=1)
        vectors.append(embeddings.squeeze().numpy())

    vectors = np.array(vectors).astype('float32')
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, transcript_vector_path)

# Example usage
# vectorize_transcript('data/transcripts/sample.txt', 'embeddings/transcript_vectors/sample.index')
