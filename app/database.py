# import json
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer

# # Load product reviews
# with open("data/reviews.json", "r") as f:
#     product_reviews = json.load(f)

# # Initialize sentence transformer
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Create embeddings
# embeddings = []
# for review in product_reviews:
#     embedding = model.encode(review['text'])
#     embeddings.append(embedding)

# # Create FAISS index
# dimension = len(embeddings[0])
# index = faiss.IndexFlatL2(dimension)
# index.add(np.array(embeddings).astype('float32'))

# def get_index():
#     return index

# def get_product_reviews():
#     return product_reviews


import json
import faiss
import numpy as np
from .embeddings import generate_embedding

# Load product reviews
with open("data/reviews.json", "r") as f:
    product_reviews = json.load(f)

# Create embeddings
embeddings = []
for review in product_reviews:
    embedding = generate_embedding(review['text'])
    embeddings.append(embedding)

embeddings = np.array(embeddings).astype('float32')

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def get_index():
    return index

def get_product_reviews():
    return product_reviews