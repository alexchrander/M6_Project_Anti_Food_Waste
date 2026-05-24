"""
Shared Gemini embedding functions.
Imported by build_index.py and query.py.
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"

_client = genai.Client(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))


def embed_recipe(text: str) -> list[float]:
    """Embed a recipe text using RETRIEVAL_DOCUMENT task type."""
    result = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return result.embeddings[0].values


def embed_query(text: str) -> list[float]:
    """Embed a user query using RETRIEVAL_QUERY task type."""
    result = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values
