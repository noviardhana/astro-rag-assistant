"""
config.py — Centralized configuration for AstroRAG.

Loads environment variables and initializes shared singletons:
    - embed_model   : SentenceTransformer for query & chunk embedding
    - gemini_client : Google GenAI client for answer generation
    - GEMINI_MODEL  : Active Gemini model name

Usage:
    from config import embed_model, gemini_client, GEMINI_MODEL
"""

import os
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv()

# ─────────────────────────────────────────────
# Environment Variables
# ─────────────────────────────────────────────
NASA_ADS_TOKEN = os.getenv("NASA_ADS_API")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = os.getenv("MODEL_GEMINI", "gemini-2.5-flash")

if not NASA_ADS_TOKEN:
    raise EnvironmentError("❌ NASA_ADS_API token not found. Check your .env file.")
if not GEMINI_API_KEY:
    raise EnvironmentError("❌ GEMINI_API_KEY not found. Check your .env file.")

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Using device: {_DEVICE.upper()}")

# ─────────────────────────────────────────────
# Embedding Model
# ─────────────────────────────────────────────
_EMBED_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"


def get_embed_model(model_name: str = _EMBED_MODEL_NAME) -> SentenceTransformer:
    """
    Load a SentenceTransformer model onto the best available device.

    Args:
        model_name (str): HuggingFace model name or local path.
                          Default: 'multi-qa-MiniLM-L6-cos-v1'
                          Alternatives:
                            - 'intfloat/e5-base-v2'         (balanced)
                            - 'BAAI/bge-large-en-v1.5'      (high accuracy)
                            - 'nomic-ai/nomic-embed-text-v1.5' (latest)

    Returns:
        SentenceTransformer: Loaded model instance.
    """
    print(f"🔧 Loading embed model '{model_name}' on {_DEVICE.upper()}...")
    return SentenceTransformer(model_name, device=_DEVICE)


# Singleton — loaded once, reused across all modules
embed_model: SentenceTransformer = get_embed_model()

# ─────────────────────────────────────────────
# Gemini Client
# ─────────────────────────────────────────────
gemini_client: genai.Client = genai.Client(api_key=GEMINI_API_KEY)
print(f"✅ Gemini client initialized — model: {GEMINI_MODEL}")
