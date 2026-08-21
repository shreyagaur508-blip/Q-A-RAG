from embedding import create_embeddings
from vector_store import create_vector_store


texts = [
    "Machine learning is a branch of artificial intelligence.",
    "Supervised learning uses labeled data.",
    "Unsupervised learning uses unlabeled data.",
    "Reinforcement learning uses rewards."
]


print("Creating embeddings...")

embeddings = create_embeddings(texts)

print("Embeddings created.")

print("Creating FAISS index...")

index = create_vector_store(embeddings)

print("FAISS index created!")

print("Number of stored vectors:", index.ntotal)