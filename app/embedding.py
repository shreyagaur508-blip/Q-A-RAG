import ollama


def create_embeddings(chunks):
    response = ollama.embed(
        model="nomic-embed-text",
        input=chunks
    )

    return response["embeddings"]