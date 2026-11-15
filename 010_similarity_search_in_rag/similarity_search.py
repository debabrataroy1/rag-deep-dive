import math

# Query embedding
A = [1, 2]

# Document embeddings
documents = {
    "Document 1": [2, 4],
    "Document 2": [4, 2],
    "Document 3": [10, 20]
}

def dot_product(a,b):
    return sum(x*y for x,y in zip(a,b))

def magnitude(a):
    return math.sqrt(sum(x*x for x in a))

def cosine_similarity(vec_a, vec_b):
    if (len(vec_a) != len(vec_b)):
        raise ValueError("Vectors must be of the same length.")
    
     # Calculate the dot product
    dot = dot_product(vec_a, vec_b)
    
    # Calculate the Euclidean norms (magnitudes)
    norm_a = magnitude(vec_a)
    norm_b = magnitude(vec_b)
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    
    return dot / (norm_a * norm_b)

def euclidean_distance(a, b):
    return math.sqrt(sum((x-y) ** 2 for x,y in zip(a, b)))

for name, doc in documents.items():
    
    cosine = cosine_similarity(A, doc)
    euclidean = euclidean_distance(A, doc)
    dot = dot_product(A, doc)

    print(f"\n{name}:")
    print("Cosine Similarity:", round(cosine, 4))
    print("Euclidean Distance:", round(euclidean, 4))
    print("Dot Product:", dot)