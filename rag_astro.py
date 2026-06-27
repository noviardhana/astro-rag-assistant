"""
AstroRAG — Astronomy RAG Pipeline (Optimized)

Pipeline:
    classify → embed → retrieve → rank → prompt → generate → format

Modules used:
    - config.py        : embed_model, gemini_client, GEMINI_MODEL
    - prompts.py       : _SYSTEM_INSTRUCTION, _PROMPT_TEMPLATE
"""

from __future__ import annotations
from dataclasses import dataclass
import chromadb
from google.genai.errors import ServerError

from config import embed_model, gemini_client, GEMINI_MODEL
from prompts import _SYSTEM_INSTRUCTION, _PROMPT_TEMPLATE


# ─────────────────────────────────────────────
# Query Classification
# ─────────────────────────────────────────────
_QUESTION_TYPES: dict[str, list[str]] = {
    "definition":  ["what is", "define", "apa itu", "pengertian"],
    "comparison":  ["compare", "difference", "versus", "vs", "bandingkan"],
    "process":     ["how does", "bagaimana", "how is", "mechanism", "proses"],
    "observation": ["observed", "detected", "discovered", "found", "measurement"],
    "recent":      ["latest", "recent", "terbaru", "2023", "2024", "2025", "new finding"],
}

_N_RESULTS_MAP: dict[str, int] = {
    "definition":  5,
    "comparison":  10,
    "process":     7,
    "observation": 8,
    "recent":      8,
    "general":     6,
}


def classify_query(query: str) -> str:
    """
    Classify a user query into a question type to tune retrieval depth.

    Args:
        query (str): Natural language question from the user.

    Returns:
        str: One of 'definition', 'comparison', 'process', 'observation',
             'recent', or 'general' (fallback).
    """
    q = query.lower()
    for qtype, keywords in _QUESTION_TYPES.items():
        if any(kw in q for kw in keywords):
            return qtype
    return "general"


# ─────────────────────────────────────────────
# Paper Context
# ─────────────────────────────────────────────
@dataclass
class PaperContext:
    """
    Structured representation of a single retrieved paper.

    Attributes:
        index   : Citation number used in the prompt, e.g. [1].
        bibcode : NASA ADS bibcode identifier.
        title   : Paper title.
        year    : Publication year (stored as str for display).
        pub     : Journal or publication name.
        abstract: Full abstract text used as context.
    """
    index:    int
    bibcode:  str
    title:    str
    year:     str
    pub:      str
    abstract: str

    def to_prompt_block(self) -> str:
        """Format paper as a numbered context block for the LLM prompt."""
        return (
            f"[{self.index}] {self.title} ({self.year}) — {self.pub}\n"
            f"Bibcode: {self.bibcode}\n"
            f"{self.abstract}\n"
        )

    def to_source_line(self) -> str:
        """Format paper as a single source citation line."""
        return f"[{self.index}] {self.title} | {self.pub} | {self.bibcode}"


# ─────────────────────────────────────────────
# Context Builder
# ─────────────────────────────────────────────
def build_context(results: dict) -> tuple[str, list[PaperContext]]:
    """
    Parse raw ChromaDB query results into a structured context string
    and a list of PaperContext objects sorted newest-first.

    Args:
        results (dict): Raw output from collection.query(), containing
                        'documents' and 'metadatas' keys.

    Returns:
        tuple:
            - str              : Formatted context block ready for the prompt.
            - list[PaperContext]: Sorted list of paper metadata objects.
    """
    docs  = results["documents"][0]
    metas = results["metadatas"][0]

    papers: list[PaperContext] = []
    for doc, meta in zip(docs, metas):
        papers.append(PaperContext(
            index=0,    # assigned after sort
            bibcode=meta.get("bibcode", "N/A"),
            title=meta.get("title",   "Unknown Title"),
            year=str(meta.get("year", "0")),    # cast to str — avoids int/str sort TypeError
            pub=meta.get("pub",     "Unknown Journal"),
            abstract=doc,
        ))

    # Sort newest-first; cast year to int for correct numeric comparison
    papers.sort(key=lambda p: int(p.year or 0), reverse=True)
    for i, p in enumerate(papers, 1):
        p.index = i

    context_block = "\n\n".join(p.to_prompt_block() for p in papers)
    return context_block, papers


# ─────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────
def build_prompt(query: str, context: str, question_type: str) -> str:
    """
    Fill the shared prompt template with the query, context, and question type.

    Args:
        query         (str): User's natural language question.
        context       (str): Formatted paper abstracts from build_context().
        question_type (str): Classified query type from classify_query().

    Returns:
        str: Complete, ready-to-send prompt string.
    """
    return _PROMPT_TEMPLATE.format(
        context_string=context,
        query_text=query,
    ).strip()


# ─────────────────────────────────────────────
# Main RAG Pipeline
# ─────────────────────────────────────────────
def generate_answer(query: str, collection: chromadb.Collection) -> str:
    """
    Full RAG pipeline: classify → embed → retrieve → rank → prompt → generate → format.

    Uses embed_model, gemini_client, and GEMINI_MODEL from config.py,
    and prompt templates from prompts.py.

    Args:
        query      (str)                 : The user's astronomy question.
        collection (chromadb.Collection) : ChromaDB collection of paper abstracts.

    Returns:
        str: Formatted answer with inline citations and appended source list.
             Returns an error message string if the Gemini API call fails.
    """
    if not query.strip():
        raise ValueError("Query must not be empty.")

    # 1. Classify query → tune retrieval depth
    q_type   = classify_query(query)
    n_results = _N_RESULTS_MAP[q_type]
    print(f"[RAG] Query type: '{q_type}' → retrieving {n_results} abstracts")

    # 2. Embed query using shared embed_model from config.py
    query_embedding = embed_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    # 3. Retrieve from ChromaDB
    raw_results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    print(f"[RAG] Retrieved {len(raw_results['documents'][0])} abstracts from ChromaDB")

    # 4. Build ranked context string + structured paper list
    context_str, papers = build_context(raw_results)

    # 5. Build final prompt
    prompt = build_prompt(query, context_str, q_type)

    # 6. Generate answer via Gemini (shared client from config.py)
    print(f"[RAG] Sending prompt to {GEMINI_MODEL}...")
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "system_instruction": _SYSTEM_INSTRUCTION,
                "temperature":        0.15,   # low = factual, no hallucination
                "max_output_tokens":  1024,
            },
        )
        answer = response.text.strip()
    except ServerError as e:
        print(f"❌ Gemini API error: {e}")
        return "An error occurred while generating the answer. Please try again."

    print("[RAG] Answer received.")

    # 7. Append formatted source list
    source_lines = "\n".join(p.to_source_line() for p in papers)
    return f"{answer}\n\n── Sources ──\n{source_lines}"