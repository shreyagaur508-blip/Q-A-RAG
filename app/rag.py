import json
import ollama

from embedding import create_embeddings
from vector_store import load_vector_store, search_vector_store


INDEX_PATH = "vectorstore/document.index"
CHUNKS_PATH = "vectorstore/chunks.json"


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def retrieve_relevant_chunks(question, k=3):
    # Load FAISS index
    index = load_vector_store(INDEX_PATH)

    # Load original text chunks
    chunks = load_chunks()

    # Convert question into an embedding
    question_embedding = create_embeddings([question])[0]

    # Search FAISS
    distances, indices = search_vector_store(
        index,
        question_embedding,
        k=min(k, len(chunks))
    )

    relevant_chunks = []

    for distance, index_number in zip(distances, indices):
        relevant_chunks.append({
            "text": chunks[index_number],
            "distance": float(distance)
        })

    return relevant_chunks


def generate_answer(question, relevant_chunks):

    context = "\n\n".join(
    f"Source: {chunk['source']}\n"
    f"{chunk['text']}"
    for chunk in relevant_chunks
)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, say:
"I couldn't find the answer in the provided document."

Do not make up information.

Context:
----------------
{context}
----------------

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def ask_question(question):

    relevant_chunks = retrieve_relevant_chunks(question)

    answer = generate_answer(
        question,
        relevant_chunks
    )

    return answer, relevant_chunks