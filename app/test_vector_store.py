from pdf_loader import extract_text_from_pdf
from chunker import create_chunks
from embedding import create_embeddings
from vector_store import create_vector_store, search_vector_store


pdf_path = "data/documents/machine_learning.pdf"


# Read PDF
text = extract_text_from_pdf(pdf_path)

# Create chunks
chunks = create_chunks(text)

print("Number of chunks:", len(chunks))


# Create embeddings for PDF chunks
embeddings = create_embeddings(chunks)

print("Embeddings created!")


# Create FAISS database
index = create_vector_store(embeddings)

print("FAISS vector store created!")


# Our test question
question = "What is supervised learning?"


# Create embedding for the question
question_embedding = create_embeddings([question])[0]


# Search for relevant chunks
distances, indices = search_vector_store(
    index,
    question_embedding,
    k=min(3, len(chunks))
)


print("\nQuestion:")
print(question)


print("\nRelevant chunks:")

for i, index_number in enumerate(indices):
    print("\n" + "=" * 60)
    print("RESULT", i + 1)
    print("=" * 60)

    print(chunks[index_number])

    print("\nDistance:", distances[i])