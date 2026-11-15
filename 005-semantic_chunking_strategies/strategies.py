import math
import numpy as np
import re
from sentence_transformers import SentenceTransformer


# cosine_similarity = a*b/(||a|| * ||b||)
def _cosine_similarity(vec_a, vec_b):
    if (len(vec_a) != len(vec_b)):
        raise ValueError("Vectors must be of the same length.")
    
     # Calculate the dot product
    dot_product = sum(a*b for a,b in zip(vec_a,vec_b))
    
    # Calculate the Euclidean norms (magnitudes)
    norm_a = math.sqrt(sum(a*a for a in vec_a))
    norm_b = math.sqrt(sum(b*b for b in vec_b))
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

def _cosine_similarity_using_np(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        
    return dot_product / (norm_a * norm_b)

def semantic_similarity_chunking(text, threshold=0.60, model_name="all-MiniLM-L6-v2"):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    model = SentenceTransformer(model_name)
    embedding = model.encode(sentences)
    chunks = []
    current_sentence = [sentences[0]]
    
    for i in range(1, len(sentences)):
        sim = _cosine_similarity(embedding[i-1], embedding[1])
        
        if sim < threshold:
            chunks.append(" ".join(current_sentence))
            current_sentence = [sentences[i]]
        else:
            current_sentence.append(sentences[i])
    
    if current_sentence:
        chunks.append(" ".join(current_sentence))
        
    return chunks
    
    