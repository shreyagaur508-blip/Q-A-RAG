import ollama


text = "Supervised learning uses labeled data to train a machine learning model."

response = ollama.embed(
    model="nomic-embed-text",
    input=text
)

embedding = response["embeddings"][0]

print("Embedding created successfully!")
print("Embedding dimensions:", len(embedding))
print("First 10 values:")
print(embedding[:10])