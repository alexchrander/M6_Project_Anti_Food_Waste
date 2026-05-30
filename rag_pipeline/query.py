"""
Recipe RAG pipeline — stage-based.

Each stage is a standalone function so steps can be added, swapped, or
removed without touching the rest of the pipeline.  A single orchestrator
(run_recipe_pipeline) chains them all.

Stages
------
1. retrieve_recipe_candidates   embed query → top-N recipes from ChromaDB/MongoDB
2. filter_recipes_by_time       drop recipes over a cooking-time budget
3. fetch_active_products        SQL: products on sale right now
4. fetch_ingredient_embedding   get a recipe's pre-built ingredient vector
5. search_product_candidates    cosine search in clearance_products ChromaDB
6. cross_reference_active       keep only hits present in today's SQL pool
7. assemble_llm_prompt          format recipes + products into LLM messages
8. call_llm                     Gemini API call
9. parse_llm_response           JSON → list of per-recipe ingredient strings
"""

import json
import os
import re
import sys
from pathlib import Path

import chromadb
import mysql.connector
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import embed_query  # noqa: E402

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

CHROMA_PATH         = str(Path(__file__).resolve().parent.parent / "data/chroma_db")
RECIPE_COLLECTION   = "recipes"
INGREDIENT_COLLECTION = "recipe_ingredients"
PRODUCT_COLLECTION  = "clearance_products"
PROMPT_PATH         = Path(__file__).resolve().parent / "prompt.json"

TOP_K_RECIPES  = 3
TOP_K_PRODUCTS = 10
CHROMA_PREFETCH = 50
LLM_MODEL      = "gemini-2.5-flash-lite"

EXCLUDED_PRODUCT_CATEGORIES = [
    "Færdigretter på køl",
    "Færdigretter på frost",
    "Færdigretter & supper",
    "Juice & smoothies",
    "Helse & kosttilskud",
]


# ── Clients ───────────────────────────────────────────────────────────────────

def _chroma_client():
    return chromadb.PersistentClient(
        path=CHROMA_PATH,
        tenant="default_tenant",
        database="default_database",
        settings=chromadb.Settings(anonymized_telemetry=False, allow_reset=False),
    )


def _mongo_collection():
    client = MongoClient(
        "mongodb://food_waste_mongo_user:food_waste_mongo_alex@food-waste-mongo:27017/"
    )
    return client["food_waste"]["recipes"]


def _mysql_connection():
    return mysql.connector.connect(
        host="food-waste-mysql",
        user="food_waste_mysql_user",
        password="food_waste_mysql_alex",
        database="food_waste_mysql",
    )


# ── Stage 1: retrieve recipe candidates ──────────────────────────────────────

def retrieve_recipe_candidates(
    query: str,
    chroma,
    n: int = TOP_K_RECIPES,
) -> list[dict]:
    """Embed query and return the top-N semantically matching recipes from MongoDB."""
    vector = embed_query(query)
    col = chroma.get_or_create_collection(
        RECIPE_COLLECTION,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )
    hits  = col.query(query_embeddings=[vector], n_results=n)
    slugs = hits["ids"][0]
    mongo = _mongo_collection()
    docs  = {d["_id"]: d for d in mongo.find({"_id": {"$in": slugs}})}
    return [docs[s] for s in slugs if s in docs]


# ── Stage 2: filter by cooking time ──────────────────────────────────────────

def _parse_minutes(time_str: str) -> int | None:
    """Parse Danish duration strings like '45 min', '1 time', '1 time 30 min'."""
    if not time_str:
        return None
    total = 0
    h = re.search(r"(\d+)\s*time", time_str)
    m = re.search(r"(\d+)\s*min",  time_str)
    if h:
        total += int(h.group(1)) * 60
    if m:
        total += int(m.group(1))
    return total if total > 0 else None


def filter_recipes_by_time(
    candidates: list[dict],
    max_minutes: int | None,
    keep: int = TOP_K_RECIPES,
) -> list[dict]:
    """Return at most `keep` recipes within the time budget (None = no limit)."""
    if max_minutes is None:
        return candidates[:keep]
    return [
        r for r in candidates
        if (t := _parse_minutes(r.get("total_time", ""))) is None or t <= max_minutes
    ][:keep]


# ── Stage 3: fetch today's active products from SQL ──────────────────────────

def fetch_active_products() -> dict[str, dict]:
    """Return all currently on-sale products keyed by EAN string."""
    conn   = _mysql_connection()
    cursor = conn.cursor(dictionary=True)
    placeholders = ", ".join(["%s"] * len(EXCLUDED_PRODUCT_CATEGORIES))
    cursor.execute(
        f"""
        SELECT product_ean, product_description,
               offer_new_price, offer_original_price, offer_percent_discount,
               store_name, store_brand, store_city, store_lat, store_lng,
               category_level1_da, category_level2_da, offer_end_time
        FROM app
        WHERE offer_end_time > NOW()
          AND category_level2_da NOT IN ({placeholders})
        """,
        EXCLUDED_PRODUCT_CATEGORIES,
    )
    rows = cursor.fetchall()
    conn.close()
    return {str(row["product_ean"]): row for row in rows}


# ── Stage 4: fetch a recipe's pre-computed ingredient embedding ───────────────

def fetch_ingredient_embedding(recipe_id: str, chroma) -> list[float] | None:
    """Return the stored ingredient vector for a recipe, or None if missing."""
    col    = chroma.get_or_create_collection(
        INGREDIENT_COLLECTION,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )
    result = col.get(ids=[recipe_id], include=["embeddings"])
    if not result["ids"]:
        return None
    return result["embeddings"][0]


# ── Stage 5: cosine search in clearance_products ChromaDB ────────────────────

def search_product_candidates(
    embedding: list[float],
    chroma,
    n: int = CHROMA_PREFETCH,
) -> list[tuple[str, float]]:
    """Return (ean, cosine_distance) pairs for the nearest clearance products."""
    col  = chroma.get_or_create_collection(
        PRODUCT_COLLECTION,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )
    hits = col.query(
        query_embeddings=[embedding],
        n_results=n,
        include=["distances"],
    )
    return list(zip(hits["ids"][0], hits["distances"][0]))


# ── Stage 6: cross-reference with today's active SQL products ─────────────────

def cross_reference_active_products(
    candidates: list[tuple[str, float]],
    active_products: dict[str, dict],
    top_k: int = TOP_K_PRODUCTS,
) -> list[tuple[float, dict]]:
    """Keep only candidates present in today's active-product pool.

    Returns (similarity_score, product_row) pairs, similarity = 1 - distance.
    """
    matched = []
    for ean, distance in candidates:
        if ean in active_products:
            matched.append((1 - distance, active_products[ean]))
        if len(matched) == top_k:
            break
    return matched


# ── Stages 4-6 combined ───────────────────────────────────────────────────────

def find_products_for_recipe(
    recipe_id: str,
    active_products: dict[str, dict],
    chroma,
) -> list[tuple[float, dict]]:
    """Run stages 4→5→6: ingredient embedding → product search → active filter."""
    embedding = fetch_ingredient_embedding(recipe_id, chroma)
    if embedding is None:
        return []
    candidates = search_product_candidates(embedding, chroma)
    return cross_reference_active_products(candidates, active_products)


# ── Stage 7: assemble LLM prompt ─────────────────────────────────────────────

def _format_recipes_for_llm(recipes: list[dict]) -> str:
    parts = []
    for i, r in enumerate(recipes, start=1):
        parts.append(
            f"Opskrift {i}: {r.get('title', '')}\n"
            f"Portioner: {r.get('servings', '')}\n"
            f"Ingredienser: {', '.join(r.get('ingredients', []))}"
        )
    return "\n\n".join(parts)


def _format_products_for_llm(
    recipes: list[dict],
    products_per_recipe: list[list[tuple[float, dict]]],
) -> str:
    sections = []
    for i, (recipe, products) in enumerate(zip(recipes, products_per_recipe), start=1):
        title = recipe.get("title", f"Opskrift {i}")
        if not products:
            sections.append(
                f"Tilbud til opskrift {i} ({title}):\n  (ingen matchende tilbud fundet)"
            )
            continue
        lines = [f"Tilbud til opskrift {i} ({title}):"]
        for similarity, p in products:
            end = p["offer_end_time"].strftime("%d. %b") if p.get("offer_end_time") else "?"
            lines.append(
                f"  - {p['product_description']} | {p['category_level2_da']} | "
                f"{p['offer_new_price']:.2f} kr ({p['offer_percent_discount']:.0f}% rabat) | "
                f"{p['store_name']}, {p['store_city']} | Tilbud slutter: {end} "
                f"[match: {similarity:.2f}]"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def assemble_llm_prompt(
    query: str,
    recipes: list[dict],
    products_per_recipe: list[list[tuple[float, dict]]],
) -> tuple[str, str]:
    """Return (system_prompt, user_message) ready for call_llm()."""
    with open(PROMPT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    active = data["prompts"][data["active"]]
    system = active["system"]
    user   = active["user_template"].format(
        query    = query,
        recipes  = _format_recipes_for_llm(recipes),
        products = _format_products_for_llm(recipes, products_per_recipe),
    )
    return system, user


# ── Stage 8: call LLM ─────────────────────────────────────────────────────────

def call_llm(system: str, user: str) -> str:
    client   = genai.Client(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
    response = client.models.generate_content(
        model    = LLM_MODEL,
        contents = user,
        config   = types.GenerateContentConfig(
            system_instruction  = system,
            response_mime_type  = "application/json",
        ),
    )
    return response.text


# ── Stage 9: parse LLM response ───────────────────────────────────────────────

def parse_llm_response(raw: str) -> list[str]:
    """Parse LLM JSON into a list of ingredient strings, one per recipe."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            sections = [
                str(data.get("opskrift_1", "")).strip(),
                str(data.get("opskrift_2", "")).strip(),
                str(data.get("opskrift_3", "")).strip(),
            ]
        elif isinstance(data, list):
            sections = [str(s).strip() for s in data[:3]]
        else:
            sections = []
    except (json.JSONDecodeError, TypeError):
        parts    = re.split(r"={2,}\s*OPSKRIFT[_\s]?\d+\s*={2,}", raw, flags=re.IGNORECASE)
        sections = [p.strip() for p in parts[1:4]]
    while len(sections) < 3:
        sections.append("")
    return sections


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_recipe_pipeline(
    query: str,
    chroma,
    max_minutes: int | None = None,
    active_products: dict[str, dict] | None = None,
) -> tuple[list[dict], list[str]]:
    """Run stages 1-9 and return (recipes, llm_sections).

    recipes       list of recipe dicts from MongoDB (for UI display)
    llm_sections  list of ingredient strings from the LLM (one per recipe)

    Pass active_products to skip Stage 3 (useful when the caller caches it).
    """
    n_candidates = TOP_K_RECIPES * 4 if max_minutes else TOP_K_RECIPES

    # Stage 1
    candidates = retrieve_recipe_candidates(query, chroma, n=n_candidates)

    # Stage 2
    recipes = filter_recipes_by_time(candidates, max_minutes)
    if not recipes:
        return [], []

    # Stage 3 (skipped if caller provides pre-fetched products)
    if active_products is None:
        active_products = fetch_active_products()

    # Stages 4-6
    products_per_recipe = [
        find_products_for_recipe(r["_id"], active_products, chroma)
        for r in recipes
    ]

    # Stage 7
    system, user_message = assemble_llm_prompt(query, recipes, products_per_recipe)

    # Stage 8
    raw = call_llm(system, user_message)

    # Stage 9
    sections = parse_llm_response(raw)

    return recipes, sections


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python rag_pipeline/query.py "<your query>"')
        sys.exit(1)

    user_query = sys.argv[1]
    print(f"Query: {user_query}\n")

    chroma = _chroma_client()
    recipes, sections = run_recipe_pipeline(user_query, chroma)

    for recipe, section in zip(recipes, sections):
        print(f"\n{'=' * 60}")
        print(f"Opskrift: {recipe.get('title')}")
        print(section)


if __name__ == "__main__":
    main()
