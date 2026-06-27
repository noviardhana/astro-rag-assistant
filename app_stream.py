import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import chromadb
from rag_astro import generate_answer

# ─────────────────────────────────────────────
# ChromaDB
# ─────────────────────────────────────────────
@st.cache_resource
def get_chroma_collection():
    client_chroma = chromadb.PersistentClient(path="./chroma_db")
    return client_chroma.get_or_create_collection(name="astro_paper")

collection = get_chroma_collection()

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AstroRAG - Astronomy Assistant",
    page_icon="🌌",
    layout="centered",
)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("🌌 AstroRAG Assistant")
st.caption("✨ Created by **Noviardhana**")
st.markdown(
    "Ask a question about astronomy, astrophysics, or cosmology. "
    "The assistant answers based on scientific papers retrieved from **NASA ADS**."
)
st.info(
    "💡 **Introduction:** This application is an experimental project built to test "
    "and learn Streamlit UI development. It was developed based on the final project "
    "for the **Sanbercode Bootcamp 2025**."
)
st.divider()

# ─────────────────────────────────────────────
# Input & Answer
# ─────────────────────────────────────────────
query = st.text_area(
    "Enter your question:",
    placeholder="e.g., What is the role of dark matter in galaxy formation?",
    height=100,
)

if st.button("🚀 Get Answer", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Retrieving papers and generating answer... ⏳"):
            try:
                answer = generate_answer(query, collection)
                st.success("Answer generated successfully!")
                st.subheader("Answer:")
                st.markdown(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")