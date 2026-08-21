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
st.caption(
    "Ask questions about your PDFs using "
    "local AI — no paid API required."
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "messages" not in st.session_state:
    st.session_state.messages = []


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

    process_button = st.button(
        "⚙️ Process Documents",
        use_container_width=True
    )

    if process_button:

        if not uploaded_files:

            st.warning(
                "Please upload at least one PDF."
            )

        else:

            all_chunks = []

            progress = st.progress(0)

            for number, uploaded_file in enumerate(
                uploaded_files
            ):

                st.write(
                    f"Reading `{uploaded_file.name}`..."
                )

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    pdf_path = temp_file.name

                # Extract PDF text
                text = extract_text_from_pdf(
                    pdf_path
                )

                if not text.strip():

                    st.warning(
                        f"No text found in "
                        f"`{uploaded_file.name}`."
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

            if not all_chunks:

                st.error(
                    "No readable text was found."
                )

            else:

                st.write(
                    f"Created {len(all_chunks)} chunks."
                )

                # Create embeddings
                with st.spinner(
                    "Creating embeddings..."
                ):

                    texts = [
                        chunk["text"]
                        for chunk in all_chunks
                    ]

                    embeddings = create_embeddings(
                        texts
                    )

                # Create FAISS index
                with st.spinner(
                    "Building vector database..."
                ):

                    index = create_vector_store(
                        embeddings
                    )

                # Store in session
                st.session_state.index = index

                st.session_state.chunks = all_chunks

                st.session_state.messages = []

                st.success(
                    "✅ Documents ready!"
                )


    st.divider()


    # --------------------------------------------------
    # DOCUMENT STATUS
    # --------------------------------------------------

    st.subheader("Status")

    if st.session_state.index is not None:

        st.success("🟢 Documents loaded")

        st.write(
            f"Chunks: "
            f"**{len(st.session_state.chunks)}**"
        )

    else:

        st.info(
            "Upload and process a document."
        )


    st.divider()


    # --------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    # Check if documents exist
    if st.session_state.index is None:

        st.warning(
            "Please upload and process a PDF first."
        )

        st.stop()


    # --------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching documents..."
        ):

            # Embed question
            question_embedding = (
                create_embeddings(
                    [question]
                )[0]
            )

            # Search FAISS
            distances, indices = (
                st.session_state.index.search(
                    np.array(
                        [question_embedding],
                        dtype="float32"
                    ),
                    min(
                        3,
                        len(
                            st.session_state.chunks
                        )
                    )
                )
            )


            # Get relevant chunks
            relevant_chunks = []

            for distance, index_number in zip(
                distances[0],
                indices[0]
            ):

                chunk = (
                    st.session_state.chunks[
                        index_number
                    ]
                )

                relevant_chunks.append({
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "distance": float(distance)
                })


        # --------------------------------------------------
        # GENERATION
        # --------------------------------------------------

        with st.spinner(
            "🤖 Generating answer..."
        ):

            answer = generate_answer(
                question,
                relevant_chunks
            )


        # Display answer
        st.markdown(answer)


        # --------------------------------------------------
        # SOURCES
        # --------------------------------------------------

        with st.expander(
            "📖 View retrieved sources"
        ):

            for number, chunk in enumerate(
                relevant_chunks
            ):

                st.markdown(
                    f"**Source {number + 1}: "
                    f"{chunk['source']}**"
                )

                st.write(
                    chunk["text"]
                )

                st.caption(
                    f"Similarity distance: "
                    f"{chunk['distance']:.4f}"
                )


    # --------------------------------------------------
    # SAVE ANSWER
    # --------------------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })