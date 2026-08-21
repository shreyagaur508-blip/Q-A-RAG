import ollama


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
"I could not find the answer in the provided documents."

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