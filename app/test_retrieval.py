import numpy as np

from embedding import create_embeddings
from vector_store import create_vector_store


# Sample document chunks
chunks = [
    "Machine learning is a branch of artificial intelligence.",
    "Supervised learning uses labeled data to train a model.",
    "Unsupervised learning works with unlabeled data.",
    "Clustering is a common example of unsupervised learning.",
    "Reinforcement learning allows an agent to learn using rewards."
]


print("Creating embeddings...")

embeddings = create_embeddings(chunks)

print("Creating FAISS index...")

index = create_vector_store(embeddings)

print("FAISS index contains:", index.ntotal, "vectors")


# --------------------------------------------------
# QUESTION
# --------------------------------------------------

question = "What type of learning uses labeled data?"

print("\nQuestion:")
print(question)


# Create question embedding
question_embedding = create_embeddings(
    [question]
)[0]


# Search the vector database
distances, indices = index.search(
    np.array(
        [question_embedding],
        dtype="float32"
    ),
    2
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\nMost relevant chunks:\n")

for distance, index_number in zip(
    distances[0],
    indices[0]
):

    print(
        f"Distance: {distance:.4f}"
    )

    print(
        chunks[index_number]
    )

    print()