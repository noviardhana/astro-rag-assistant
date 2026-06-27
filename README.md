# 🌌 AstroRAG — Astronomy Research Assistant

> A Retrieval-Augmented Generation (RAG) system for navigating and querying scientific literature from the NASA Astrophysics Data System (ADS).

---

## Overview

The volume of astronomy publications grows faster than any researcher can manually review. **AstroRAG** addresses this challenge by combining semantic search with large language model generation to deliver precise, citation-grounded answers from peer-reviewed abstracts.

Built on top of **NASA ADS**, the system automatically retrieves the most relevant papers for a given question and synthesises a structured scientific response — without hallucination, without speculation.

### Key Capabilities

- **Question Answering** — ask natural language questions; get answers grounded in real paper abstracts
- **Adaptive Retrieval** — query type is automatically classified (definition, comparison, process, observation, recent) to tune retrieval depth
- **Inline Citations** — every factual claim is cited with its source paper `[1]`, `[2,4]`
- **Multi-topic Coverage** — cosmology, stellar physics, solar system, galactic astronomy, high-energy astrophysics

---

## Architecture

```
User Query
    │
    ▼
classify_query()          # detect question type → tune n_results
    │
    ▼
embed_model.encode()      # SentenceTransformer (multi-qa-MiniLM-L6-cos-v1)
    │
    ▼
ChromaDB.query()          # semantic retrieval from vectorised abstracts
    │
    ▼
build_context()           # rank by recency, format numbered context blocks
    │
    ▼
_PROMPT_TEMPLATE          # structured prompt with strict grounding rules
    │
    ▼
Gemini (gemini-2.5-flash) # answer generation (temperature=0.15)
    │
    ▼
Formatted Answer + Sources
```

---

## Project Structure

```
project/
├── app_stream.py        # Streamlit UI
├── config.py            # environment variables, embed_model, gemini_client
├── prompts.py           # _SYSTEM_INSTRUCTION, _PROMPT_TEMPLATE
├── rag_astro.py         # main RAG pipeline (classify→embed→retrieve→generate)
├── query.py             # semantic search utility (query_collection, print_results)
├── notebook.ipnb        # Data retrieval through to embedding
├── requirements.txt     # Python dependencies
├── chroma_db/           # persistent ChromaDB vector store (excluded from Git)
└── .env                 # API keys (excluded from Git)
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- NASA ADS API token → [get one here](https://ui.adsabs.harvard.edu/user/settings/token)
- Google Gemini API key → [get one here](https://aistudio.google.com/app/apikey)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/astro-rag-assistant.git
cd astrorag
```

**2. Create and activate a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a `.env` file in the project root:
```env
NASA_ADS_API=your_nasa_ads_token_here
GEMINI_API_KEY=your_gemini_api_key_here
MODEL_GEMINI=gemini-2.5-flash
```

**5. Run the Streamlit app**
```bash
streamlit run app.py
```

---

## Data Pipeline (Notebook)

If you need to rebuild the vector database from scratch, run the following steps in order inside the notebook:

| Step | Module | Function |
|------|--------|----------|
| 1. Fetch papers | `data_collection` | `collect_astronomy_data()` |
| 2. Clean text | `preprocessing` | `clean_dataframe()` |
| 3. Chunk abstracts | `chunking` | `chunk_from_content()` |
| 4. Generate embeddings | `embedding` | `embed_chunks()` |
| 5. Store to ChromaDB | `chroma_store` | `store_to_chroma()` |

---

## Evaluation

Answer quality is assessed using cosine similarity between RAG-generated answers and human-written reference answers, computed via the same SentenceTransformer model used for retrieval.

```python
from embedding import embed_model
scores, avg = evaluate_rag_answers(model_answers, gold_answers, eval_model=embed_model)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Embedding | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` |
| Vector Store | ChromaDB |
| LLM | Google Gemini 2.5 Flash |
| Data Source | NASA Astrophysics Data System (ADS) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |

---

## Live Demo

🔗 [astro-rag-assistant.streamlit.app](https://astro-rag-assistant-prototipe.streamlit.app)

---

## License

This project is distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

Data retrieved from the NASA Astrophysics Data System (ADS) is subject to the
[NASA ADS Terms of Use](https://ui.adsabs.harvard.edu/help/terms/).

---

## Contact

Developed as a final project prototype for the **Sanbercode Bootcamp 2025**, exploring RAG systems applied to scientific literature.
For questions or collaboration, please reach out via GitHub Issues.
