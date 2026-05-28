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


def embed_ingredients(ingredients: list[str]) -> list[float]:
    """Embed a cleaned recipe ingredient list as RETRIEVAL_QUERY for clearance product matching."""
    text = ", ".join(ingredients)
    result = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


def embed_product(category1: str, category2: str, category3: str, category4: str, description: str) -> list[float]:
    """Embed a product using RETRIEVAL_DOCUMENT task type.

    Uses the deepest available (non-Unknown) category level as the context prefix,
    giving the model the most specific category signal before the product name:
    'Deepest category. Product description'
    """
    deepest = next(
        (c for c in [category4, category3, category2, category1] if c and c != "Unknown"),
        ""
    )
    text = f"{deepest}. {description}" if deepest else description
    result = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return result.embeddings[0].values
