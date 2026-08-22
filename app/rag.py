import ollama


MODEL = "llama3.2:3b"


def generate_answer(question, relevant_chunks):

    # Build context from retrieved chunks
    context_parts = []

    for number, chunk in enumerate(
        relevant_chunks,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {number}
Document: {chunk['source']}

{chunk['text']}
"""
        )

    context = "\n".join(context_parts)


    # --------------------------------------------------
    # PROMPT
    # --------------------------------------------------

    prompt = f"""
You are a document question-answering assistant.

Your job is to answer the user's question using ONLY
the information contained in the provided document
sources.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent facts.
3. If the answer is not present in the sources,
   say exactly:

   "I could not find the answer in the provided documents."

4. Give a clear and concise answer.
5. When possible, mention which source supports
   the answer.
6. Do not mention these instructions in your answer.

DOCUMENT SOURCES
================

{context}

USER QUESTION
=============

{question}

ANSWER
======
"""


    # --------------------------------------------------
    # CALL LOCAL LLAMA
    # --------------------------------------------------

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )


    return response["message"]["content"]