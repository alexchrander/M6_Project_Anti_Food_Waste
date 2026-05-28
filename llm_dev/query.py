"""
Recipe RAG query pipeline.

Embeds a user query, retrieves the top-3 matching recipes from ChromaDB,
then for each recipe finds the top-10 semantically aligned clearance products
from the app table using ingredient-level embeddings. The LLM receives at most
30 products (10 per recipe), all chosen for relevance.

Usage:
    python llm_dev/query.py "jeg vil gerne have noget italiensk pasta"
"""

import json
import os
import sys
from pathlib import Path

import chromadb
import mysql.connector
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import embed_query

load_dotenv()

CHROMA_PATH = "data/chroma_db"
RECIPE_COLLECTION = "recipes"
INGREDIENT_COLLECTION = "recipe_ingredients"
PRODUCT_COLLECTION = "clearance_products"
PROMPT_PATH = Path(__file__).resolve().parent / "prompt.json"
TOP_K_RECIPES = 3
TOP_K_PRODUCTS = 10       # products per recipe sent to the LLM
CHROMA_PREFETCH = 50      # candidates fetched from ChromaDB before SQL cross-reference
LLM_MODEL = "gemini-2.5-flash-lite"

# Category 2 values to exclude from product recommendations.
# These are non-ingredient categories that are irrelevant or counterproductive
# (e.g. ready-made meals defeat the purpose of ingredient-based matching).
EXCLUDED_PRODUCT_CATEGORIES = [
    "Færdigretter på køl",
    "Færdigretter på frost",
    "Færdigretter & supper",
    "Juice & smoothies",
    "Helse & kosttilskud",
]


# ── Clients ───────────────────────────────────────────────────────────────────

def _chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_mongo_collection():
    client = MongoClient(
        "mongodb://food_waste_mongo_user:food_waste_mongo_alex@food-waste-mongo:27017/"
    )
    return client["food_waste"]["recipes"]


def get_mysql_connection():
    return mysql.connector.connect(
        host="food-waste-mysql",
        user="food_waste_mysql_user",
        password="food_waste_mysql_alex",
        database="food_waste_mysql",
    )


# ── Data fetching ─────────────────────────────────────────────────────────────

def retrieve_recipes(user_query: str, n_results: int = TOP_K_RECIPES) -> list[dict]:
    """Embed user query, find top-n recipes in ChromaDB, fetch full docs from MongoDB."""
    vector = embed_query(user_query)

    chroma = _chroma_client()
    hits = chroma.get_collection(RECIPE_COLLECTION).query(
        query_embeddings=[vector], n_results=n_results
    )
    slugs = hits["ids"][0]

    mongo = get_mongo_collection()
    # Preserve ranking order from ChromaDB
    docs = {d["_id"]: d for d in mongo.find({"_id": {"$in": slugs}})}
    return [docs[s] for s in slugs if s in docs]


def fetch_current_products() -> dict[str, dict]:
    """
    Fetch all currently active clearance products from the app table,
    excluding irrelevant categories. Returns a dict keyed by EAN string
    so ChromaDB results can be cross-referenced instantly.
    """
    sql = get_mysql_connection()
    cursor = sql.cursor(dictionary=True)
    placeholders = ", ".join(["%s"] * len(EXCLUDED_PRODUCT_CATEGORIES))
    cursor.execute(f"""
        SELECT
            product_ean,
            product_description,
            offer_new_price,
            offer_original_price,
            offer_percent_discount,
            store_name,
            store_brand,
            store_city,
            category_level1_da,
            category_level2_da,
            offer_end_time
        FROM app
        WHERE offer_end_time > NOW()
          AND category_level2_da NOT IN ({placeholders})
    """, EXCLUDED_PRODUCT_CATEGORIES)
    rows = cursor.fetchall()
    sql.close()
    return {str(row["product_ean"]): row for row in rows}


def get_top_products_for_recipe(
    recipe_slug: str,
    current_products: dict[str, dict],
    chroma_client,
) -> list[tuple[float, dict]]:
    """
    Find the top-10 clearance products most semantically aligned with a recipe's
    ingredient list.

    Steps:
      1. Fetch the recipe's ingredient-only embedding from recipe_ingredients.
      2. Query clearance_products ChromaDB for the CHROMA_PREFETCH nearest neighbours.
      3. Cross-reference with current_products (active SQL rows) and keep only matches.
      4. Return up to TOP_K_PRODUCTS as (similarity_score, product_row) tuples.
    """
    ing_col = chroma_client.get_collection(INGREDIENT_COLLECTION)
    result = ing_col.get(ids=[recipe_slug], include=["embeddings"])
    if not result["ids"]:
        return []

    ingredient_vector = result["embeddings"][0]

    prod_col = chroma_client.get_collection(PRODUCT_COLLECTION)
    hits = prod_col.query(
        query_embeddings=[ingredient_vector],
        n_results=CHROMA_PREFETCH,
        include=["distances"],
    )

    matched = []
    for ean, dist in zip(hits["ids"][0], hits["distances"][0]):
        ean_str = str(ean)
        if ean_str in current_products:
            similarity = 1 - dist
            matched.append((similarity, current_products[ean_str]))
        if len(matched) == TOP_K_PRODUCTS:
            break

    return matched


# ── Formatting ────────────────────────────────────────────────────────────────

def format_recipes(recipes: list[dict]) -> str:
    """Full recipe text for CLI display (includes instructions)."""
    parts = []
    for i, r in enumerate(recipes, start=1):
        instructions = []
        for section in r.get("instructions", []):
            if section.get("section"):
                instructions.append(f"  {section['section']}:")
            for step in section.get("steps", []):
                instructions.append(f"    {step['step']}. {step['text']}")

        parts.append(
            f"Opskrift {i}: {r.get('title', '')}\n"
            f"Beskrivelse: {r.get('description', '')}\n"
            f"Ingredienser: {', '.join(r.get('ingredients', []))}\n"
            f"Fremgangsmåde:\n" + "\n".join(instructions)
        )
    return "\n\n".join(parts)


def format_recipes_for_llm(recipes: list[dict], desired_servings: int = 4) -> str:
    """Compact recipe text for LLM prompt (title, servings, ingredients only)."""
    parts = []
    for i, r in enumerate(recipes, start=1):
        original = r.get("servings", "4")
        parts.append(
            f"Opskrift {i}: {r.get('title', '')}\n"
            f"Portioner: {original} -> skal skaleres til {desired_servings} portioner\n"
            f"Ingredienser: {', '.join(r.get('ingredients', []))}"
        )
    return "\n\n".join(parts)


def format_products_per_recipe(
    recipes: list[dict],
    products_per_recipe: list[list[tuple[float, dict]]],
) -> str:
    """Format per-recipe product lists into a single prompt section."""
    sections = []
    for i, (recipe, products) in enumerate(zip(recipes, products_per_recipe), start=1):
        title = recipe.get("title", f"Opskrift {i}")
        if not products:
            sections.append(f"Tilbud til opskrift {i} ({title}):\n  (ingen matchende tilbud fundet)")
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


# ── Prompt ────────────────────────────────────────────────────────────────────

def load_prompt() -> tuple[str, str]:
    with open(PROMPT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    active = data["prompts"][data["active"]]
    return active["system"], active["user_template"]


# ── LLM call ─────────────────────────────────────────────────────────────────

def call_llm(system: str, user: str) -> str:
    client = genai.Client(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
        ),
    )
    return response.text


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python llm_dev/query.py \"<your query>\"")
        sys.exit(1)

    user_query = sys.argv[1]
    print(f"Query: {user_query}\n")

    print("Retrieving recipes...")
    recipes = retrieve_recipes(user_query)

    print("Fetching current clearance products from SQL...")
    current_products = fetch_current_products()
    print(f"  {len(current_products)} active products available.\n")

    print("Finding semantically aligned products per recipe...")
    chroma = _chroma_client()
    products_per_recipe = []
    for recipe in recipes:
        slug = recipe["_id"]
        matches = get_top_products_for_recipe(slug, current_products, chroma)
        print(f"  {recipe.get('title', slug)}: {len(matches)} products matched")
        products_per_recipe.append(matches)

    system, user_template = load_prompt()

    user_message = user_template.format(
        query=user_query,
        recipes=format_recipes(recipes),
        products=format_products_per_recipe(recipes, products_per_recipe),
    )

    print("\nCalling LLM...\n")
    answer = call_llm(system, user_message)
    print(answer)


if __name__ == "__main__":
    main()