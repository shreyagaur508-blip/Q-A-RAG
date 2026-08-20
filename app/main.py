import tempfile

import numpy as np
import streamlit as st

from pdf_loader import extract_text_from_pdf
from chunker import create_chunks
from embedding import create_embeddings
from vector_store import create_vector_store
from rag import generate_answer


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📚 Document Q&A")
st.caption("RAG-powered document assistant using local AI")


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process Documents", use_container_width=True):

        if not uploaded_files:

            st.warning("Please upload at least one PDF.")

        else:

            all_chunks = []

            progress = st.progress(0)

            for number, uploaded_file in enumerate(
                uploaded_files
            ):

                st.write(
                    f"Processing `{uploaded_file.name}`..."
                )

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    pdf_path = temp_file.name

                # Extract text
                text = extract_text_from_pdf(pdf_path)

                if not text.strip():

                    st.warning(
                        f"No readable text found in "
                        f"{uploaded_file.name}"
                    )

                    continue

                # Create chunks
                chunks = create_chunks(text)

                for chunk in chunks:

                    all_chunks.append({
                        "text": chunk,
                        "source": uploaded_file.name
                    })

                progress.progress(
                    (number + 1) / len(uploaded_files)
                )

            if all_chunks:

                chunk_texts = [
                    chunk["text"]
                    for chunk in all_chunks
                ]

                with st.spinner(
                    "Creating embeddings..."
                ):

                    embeddings = create_embeddings(
                        chunk_texts
                    )

                with st.spinner(
                    "Building vector database..."
                ):

                    index = create_vector_store(
                        embeddings
                    )

                st.session_state.index = index

                st.session_state.chunks = all_chunks

                st.session_state.messages = []

                st.success(
                    f"Processed {len(uploaded_files)} "
                    f"document(s)."
                )

                st.info(
                    f"Created {len(all_chunks)} chunks."
                )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# --------------------------------------------------
# SHOW CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask something about your documents..."
)


if question:

    # Check documents
    if st.session_state.index is None:

        st.warning(
            "Please upload and process a PDF first."
        )

        st.stop()


    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)


    # Retrieve relevant chunks
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching your documents..."
        ):

            question_embedding = create_embeddings(
                [question]
            )[0]

            distances, indices = (
                st.session_state.index.search(
                    np.array(
                        [question_embedding],
                        dtype="float32"
                    ),
                    min(
                        3,
                        len(st.session_state.chunks)
                    )
                )
            )


            relevant_chunks = []

            for distance, index_number in zip(
                distances[0],
                indices[0]
            ):

                chunk = st.session_state.chunks[
                    index_number
                ]

                relevant_chunks.append({
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "distance": float(distance)
                })


        # Generate answer
        with st.spinner(
            "Generating answer..."
        ):

            answer = generate_answer(
                question,
                relevant_chunks
            )


        st.markdown(answer)


        # Sources
        with st.expander("📖 View sources"):

            for number, source in enumerate(
                relevant_chunks
            ):

                st.markdown(
                    f"**Source {number + 1}: "
                    f"{source['source']}**"
                )

                st.write(
                    source["text"]
                )

                st.caption(
                    f"Distance: "
                    f"{source['distance']:.4f}"
                )


    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })