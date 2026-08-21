from embedding import create_embeddings


texts = [
    "Machine learning is a branch of artificial intelligence.",
    "Supervised learning uses labeled data."
]


embeddings = create_embeddings(texts)


print("Embedding test successful!")
print("Number of embeddings:", len(embeddings))
print("Embedding dimension:", len(embeddings[0]))
print("First 10 values:")
print(embeddings[0][:10])