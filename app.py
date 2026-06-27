import streamlit as st
import chromadb
from rag_astro import generate_answer_with_rag 

# ========================
# 1️⃣ Initialize ChromaDB Collection
# ========================
@st.cache_resource
def get_chroma_collection():
    """Menggunakan cache agar ChromaDB tidak diload berulang kali setiap Streamlit me-refresh UI"""
    client_chroma = chromadb.PersistentClient(
        path=r"D:\Sanbercode\otomasi\final_project\chroma_db"
    )
    return client_chroma.get_or_create_collection(name="astro_paper")

collection = get_chroma_collection()

# ========================
# 2️⃣ Streamlit UI
# ========================
st.set_page_config(
    page_title="AstroRAG - Astronomy Assistant", 
    page_icon="🌌", 
    layout="centered" # "centered" biasanya lebih nyaman dibaca untuk chatbot/tanya-jawab
)

st.title("🌌 AstroRAG Assistant")
st.markdown(
    """
    Ask a question about astronomy, astrophysics, or cosmology. 
    The assistant answers based on scientific papers retrieved from **NASA ADS**.
    """
)
st.divider()

# Input Box
query = st.text_area("Enter your question:", placeholder="e.g., What is the role of dark matter in galaxy formation?", height=100)

if st.button("🚀 Get Answer", use_container_width=True):
    if query.strip() == "":
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Retrieving papers and generating answer... Please wait ⏳"):
            try:
                # Run RAG pipeline
                answer = generate_answer_with_rag(query, collection, n_results=10)
                
                st.success("Answer generated successfully!")
                st.subheader("Answer:")
                
                # Menggunakan st.markdown agar formatting seperti list dan bold dari LLM tampil rapi
                st.markdown(answer)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
