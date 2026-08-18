from pdf_loader import extract_text_from_pdf
from chunker import create_chunks
from embedding import create_embeddings


pdf_path = "data/documents/machine_learning.pdf"


# 1. Extract text
text = extract_text_from_pdf(pdf_path)

print("Text extracted successfully!")


# 2. Create chunks
chunks = create_chunks(text)

print("Number of chunks:", len(chunks))


# 3. Create embeddings
embeddings = create_embeddings(chunks)

print("Embeddings created successfully!")
print("Number of embeddings:", len(embeddings))
print("Embedding dimensions:", len(embeddings[0]))

print("\nFirst 10 values:")
print(embeddings[0][:10])