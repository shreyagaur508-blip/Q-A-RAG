import ollama


MODEL = "nomic-embed-text"


def create_embeddings(texts, batch_size=32):
    """
    Create embeddings using Ollama in batches.
    """

    all_embeddings = []

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i + batch_size]

        response = ollama.embed(
            model=MODEL,
            input=batch
        )

        all_embeddings.extend(
            response["embeddings"]
        )

    return all_embeddings