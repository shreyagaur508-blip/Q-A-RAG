import faiss
import numpy as np


def create_vector_store(embeddings):
    embeddings = np.array(embeddings, dtype="float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def save_vector_store(index, path):
    faiss.write_index(index, path)


def load_vector_store(path):
    return faiss.read_index(path)


def search_vector_store(index, query_embedding, k=3):
    query_embedding = np.array(
        [query_embedding],
        dtype="float32"
    )

    distances, indices = index.search(
        query_embedding,
        k
    )

    return distances[0], indices[0]