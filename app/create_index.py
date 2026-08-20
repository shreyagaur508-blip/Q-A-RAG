from pdf_loader import extract_text_from_pdf
from chunker import create_chunks
from embedding import create_embeddings
from vector_store import create_vector_store, save_vector_store
import json


pdf_path = "data/documents/machine_learning.pdf"
index_path = "vectorstore/document.index"


# Extract PDF text
text = extract_text_from_pdf(pdf_path)

print("PDF text extracted.")


# Create chunks
chunks = create_chunks(text)

print("Number of chunks:", len(chunks))


# Create embeddings
embeddings = create_embeddings(chunks)

print("Embeddings created.")


# Create FAISS index
index = create_vector_store(embeddings)

print("FAISS index created.")


# Save index
save_vector_store(index, index_path)
with open("vectorstore/chunks.json", "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)

print("Chunks saved successfully!")

print("FAISS index saved successfully!")