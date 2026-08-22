import os
import pickle

import faiss
import numpy as np


INDEX_DIR = "data/index"

INDEX_PATH = os.path.join(
    INDEX_DIR,
    "faiss.index"
)

CHUNKS_PATH = os.path.join(
    INDEX_DIR,
    "chunks.pkl"
)

DOCUMENTS_PATH = os.path.join(
    INDEX_DIR,
    "documents.pkl"
)


def create_vector_store(embeddings, chunks, documents):

    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    os.makedirs(
        INDEX_DIR,
        exist_ok=True
    )

    # Save FAISS index
    faiss.write_index(
        index,
        INDEX_PATH
    )

    # Save chunks
    with open(CHUNKS_PATH, "wb") as file:

        pickle.dump(
            chunks,
            file
        )

    # Save document names
    with open(DOCUMENTS_PATH, "wb") as file:

        pickle.dump(
            documents,
            file
        )

    return index


def load_vector_store():

    if not os.path.exists(INDEX_PATH):

        return None

    return faiss.read_index(
        INDEX_PATH
    )


def load_chunks():

    if not os.path.exists(CHUNKS_PATH):

        return None

    with open(
        CHUNKS_PATH,
        "rb"
    ) as file:

        return pickle.load(file)


def load_documents():

    if not os.path.exists(DOCUMENTS_PATH):

        return []

    with open(
        DOCUMENTS_PATH,
        "rb"
    ) as file:

        return pickle.load(file)