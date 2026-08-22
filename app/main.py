import os
import tempfile

import numpy as np
import streamlit as st

from pdf_loader import extract_text_from_pdf
from chunker import create_chunks
from embedding import create_embeddings

from vector_store import (
    create_vector_store,
    load_vector_store,
    load_chunks,
    load_documents,
)

from rag import generate_answer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Document Q&A | Local RAG",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("📚 Document Q&A")

st.markdown(
    "Upload PDF documents and ask questions about their content "
    "using a completely local RAG pipeline."
)

st.caption(
    "Powered by Ollama • Llama 3.2 • nomic-embed-text • FAISS"
)


# ============================================================
# SESSION STATE
# ============================================================

if "index" not in st.session_state:
    st.session_state.index = load_vector_store()

if "chunks" not in st.session_state:
    st.session_state.chunks = load_chunks() or []

if "documents" not in st.session_state:
    st.session_state.documents = load_documents()

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Document Manager")

    # --------------------------------------------------------
    # UPLOAD PDFS
    # --------------------------------------------------------

    uploaded_files = st.file_uploader(
        "Upload PDF document(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload one or multiple PDF files.",
    )

    # --------------------------------------------------------
    # PROCESS DOCUMENTS
    # --------------------------------------------------------

    if st.button(
        "⚙️ Process Documents",
        use_container_width=True,
        type="primary",
    ):

        if not uploaded_files:

            st.warning(
                "Please upload at least one PDF document."
            )

        else:

            all_chunks = []
            documents = []

            progress_bar = st.progress(0)

            status_text = st.empty()

            # ------------------------------------------------
            # PROCESS EACH PDF
            # ------------------------------------------------

            for file_number, uploaded_file in enumerate(
                uploaded_files
            ):

                document_name = uploaded_file.name

                documents.append(document_name)

                status_text.write(
                    f"📖 Reading `{document_name}`..."
                )

                # --------------------------------------------
                # CREATE TEMPORARY PDF
                # --------------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    pdf_path = temp_file.name

                try:

                    # ----------------------------------------
                    # EXTRACT TEXT
                    # ----------------------------------------

                    text = extract_text_from_pdf(
                        pdf_path
                    )

                finally:

                    # ----------------------------------------
                    # DELETE TEMPORARY FILE
                    # ----------------------------------------

                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)

                # --------------------------------------------
                # CHECK TEXT
                # --------------------------------------------

                if not text or not text.strip():

                    st.warning(
                        f"⚠️ No readable text found in "
                        f"`{document_name}`."
                    )

                    progress_bar.progress(
                        (file_number + 1)
                        / len(uploaded_files)
                    )

                    continue

                # --------------------------------------------
                # CREATE CHUNKS
                # --------------------------------------------

                status_text.write(
                    f"✂️ Splitting `{document_name}` into chunks..."
                )

                chunks = create_chunks(text)

                # --------------------------------------------
                # ADD SOURCE INFORMATION
                # --------------------------------------------

                for chunk in chunks:

                    all_chunks.append(
                        {
                            "text": chunk,
                            "source": document_name,
                        }
                    )

                progress_bar.progress(
                    (file_number + 1)
                    / len(uploaded_files)
                )

            # ------------------------------------------------
            # CHECK CHUNKS
            # ------------------------------------------------

            if not all_chunks:

                progress_bar.empty()
                status_text.empty()

                st.error(
                    "❌ No usable text was found "
                    "in the uploaded PDF(s)."
                )

            else:

                # --------------------------------------------
                # EMBEDDINGS
                # --------------------------------------------

                status_text.write(
                    "🧠 Creating document embeddings..."
                )

                with st.spinner(
                    "Creating embeddings with "
                    "nomic-embed-text..."
                ):

                    texts = [
                        chunk["text"]
                        for chunk in all_chunks
                    ]

                    embeddings = create_embeddings(
                        texts
                    )

                # --------------------------------------------
                # VECTOR STORE
                # --------------------------------------------

                status_text.write(
                    "🗂️ Building FAISS vector database..."
                )

                with st.spinner(
                    "Building FAISS vector database..."
                ):

                    index = create_vector_store(
                        embeddings,
                        all_chunks,
                        documents,
                    )

                # --------------------------------------------
                # UPDATE SESSION STATE
                # --------------------------------------------

                st.session_state.index = index

                st.session_state.chunks = all_chunks

                st.session_state.documents = documents

                st.session_state.messages = []

                progress_bar.empty()
                status_text.empty()

                st.success(
                    f"✅ Successfully processed "
                    f"{len(documents)} document(s) "
                    f"and created {len(all_chunks)} chunks."
                )

    st.divider()

    # ========================================================
    # KNOWLEDGE BASE
    # ========================================================

    st.subheader("📊 Knowledge Base")

    if st.session_state.index is not None:

        st.success("🟢 Knowledge base loaded")

        st.write(
            f"**Documents:** "
            f"{len(st.session_state.documents)}"
        )

        if st.session_state.documents:

            for document in st.session_state.documents:

                st.write(
                    f"📄 {document}"
                )

        st.write(
            f"**Chunks:** "
            f"{len(st.session_state.chunks)}"
        )

        st.write(
            "**Embedding model:** "
            "`nomic-embed-text`"
        )

        st.write(
            "**Language model:** "
            "`llama3.2:3b`"
        )

        st.write(
            "**Vector database:** "
            "`FAISS`"
        )

    else:

        st.info(
            "No documents have been processed yet."
        )

    st.divider()

    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()

    # ========================================================
    # CLEAR KNOWLEDGE BASE
    # ========================================================

    if st.button(
        "🧹 Clear Knowledge Base",
        use_container_width=True,
    ):

        files_to_delete = [
            "data/index/faiss.index",
            "data/index/chunks.pkl",
            "data/index/documents.pkl",
        ]

        for file_path in files_to_delete:

            if os.path.exists(file_path):

                try:
                    os.remove(file_path)

                except PermissionError:

                    st.error(
                        f"Could not delete `{file_path}`. "
                        "Please close any program using the file."
                    )

        st.session_state.index = None

        st.session_state.chunks = []

        st.session_state.documents = []

        st.session_state.messages = []

        st.success(
            "🧹 Knowledge base cleared successfully."
        )

        st.rerun()


# ============================================================
# WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    st.info(
        "👋 Upload a PDF from the sidebar, process it, "
        "and then ask a question below."
    )


# ============================================================
# DISPLAY PREVIOUS CHAT
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # -----------------------------------------------
        # DISPLAY SOURCES FOR ASSISTANT MESSAGES
        # -----------------------------------------------

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📖 View retrieved sources"
            ):

                for number, source in enumerate(
                    message["sources"]
                ):

                    st.markdown(
                        f"### Source {number + 1}"
                    )

                    st.caption(
                        f"📄 {source['source']}"
                    )

                    st.write(
                        source["text"]
                    )

                    st.caption(
                        f"FAISS distance: "
                        f"{source['distance']:.4f}"
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    # ========================================================
    # CHECK KNOWLEDGE BASE
    # ========================================================

    if st.session_state.index is None:

        st.warning(
            "⚠️ Please upload and process a PDF first."
        )

        st.stop()

    if not st.session_state.chunks:

        st.warning(
            "⚠️ No document chunks are available."
        )

        st.stop()

    # ========================================================
    # DISPLAY USER MESSAGE
    # ========================================================

    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # ========================================================
    # CREATE QUESTION EMBEDDING
    # ========================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching your documents..."
        ):

            question_embedding = create_embeddings(
                [question]
            )[0]

        # ====================================================
        # FAISS SEARCH
        # ====================================================

        number_of_results = min(
            3,
            len(st.session_state.chunks),
        )

        with st.spinner(
            "📚 Finding relevant information..."
        ):

            distances, indices = (
                st.session_state.index.search(
                    np.array(
                        [question_embedding],
                        dtype="float32",
                    ),
                    number_of_results,
                )
            )

        # ====================================================
        # COLLECT RELEVANT CHUNKS
        # ====================================================

        relevant_chunks = []

        for distance, chunk_index in zip(
            distances[0],
            indices[0],
        ):

            if chunk_index < 0:
                continue

            if chunk_index >= len(
                st.session_state.chunks
            ):
                continue

            chunk = (
                st.session_state.chunks[
                    chunk_index
                ]
            )

            # --------------------------------------------
            # SUPPORT BOTH DICTIONARY AND STRING CHUNKS
            # --------------------------------------------

            if isinstance(chunk, dict):

                chunk_text = chunk.get(
                    "text",
                    "",
                )

                source = chunk.get(
                    "source",
                    "Unknown document",
                )

            else:

                chunk_text = str(chunk)

                source = "Unknown document"

            relevant_chunks.append(
                {
                    "text": chunk_text,
                    "source": source,
                    "distance": float(
                        distance
                    ),
                }
            )

        # ====================================================
        # CHECK RETRIEVAL
        # ====================================================

        if not relevant_chunks:

            answer = (
                "I could not find relevant information "
                "in the uploaded documents."
            )

        else:

            # =================================================
            # GENERATE ANSWER WITH LOCAL LLAMA
            # =================================================

            with st.spinner(
                "🤖 Generating answer with Llama 3.2..."
            ):

                answer = generate_answer(
                    question,
                    relevant_chunks,
                )

        # ====================================================
        # DISPLAY ANSWER
        # ====================================================

        st.markdown(answer)

        # ====================================================
        # DISPLAY SOURCES
        # ====================================================

        if relevant_chunks:

            with st.expander(
                "📖 View retrieved sources"
            ):

                for number, chunk in enumerate(
                    relevant_chunks
                ):

                    st.markdown(
                        f"### Source {number + 1}"
                    )

                    st.caption(
                        f"📄 {chunk['source']}"
                    )

                    st.write(
                        chunk["text"]
                    )

                    st.caption(
                        f"FAISS distance: "
                        f"{chunk['distance']:.4f}"
                    )

    # ========================================================
    # SAVE ASSISTANT MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": relevant_chunks,
        }
    )