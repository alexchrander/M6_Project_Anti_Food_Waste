"""
Recipe RAG query pipeline.

Embeds a user query, retrieves the top-3 matching recipes from ChromaDB,
fetches full recipe docs from MongoDB, fetches current Salling discounted
products from MySQL, builds a prompt and sends it to the LLM via OpenRouter.

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
CHROMA_COLLECTION = "recipes"
PROMPT_PATH = Path(__file__).resolve().parent / "prompt.json"
TOP_K = 3
LLM_MODEL = "gemini-2.5-flash-lite"


# ── Clients ───────────────────────────────────────────────────────────────────

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(CHROMA_COLLECTION)


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

def retrieve_recipes(user_query: str) -> list[dict]:
    """Embed query, search ChromaDB, fetch full docs from MongoDB."""
    vector = embed_query(user_query)

    chroma = get_chroma_collection()
    hits = chroma.query(query_embeddings=[vector], n_results=TOP_K)
    slugs = hits["ids"][0]

    mongo = get_mongo_collection()
    return list(mongo.find({"_id": {"$in": slugs}}))


def fetch_products() -> list[dict]:
    """Fetch current discounted products from MySQL app table."""
    sql = get_mysql_connection()
    cursor = sql.cursor(dictionary=True)
    cursor.execute("""
        SELECT
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
        ORDER BY offer_percent_discount DESC
    """)
    products = cursor.fetchall()
    sql.close()
    return products


# ── Formatting ────────────────────────────────────────────────────────────────

def format_recipes(recipes: list[dict]) -> str:
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


def format_products(products: list[dict]) -> str:
    lines = []
    for p in products:
        end = p["offer_end_time"].strftime("%d. %b") if p.get("offer_end_time") else "?"
        lines.append(
            f"- {p['product_description']} | {p['category_level2_da']} | "
            f"{p['offer_new_price']:.2f} kr ({p['offer_percent_discount']:.0f}% rabat) | "
            f"{p['store_name']}, {p['store_city']} | Tilbud slutter: {end}"
        )
    return "\n".join(lines)


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
        config=types.GenerateContentConfig(system_instruction=system),
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

    print("Fetching products...")
    products = fetch_products()

    system, user_template = load_prompt()

    user_message = user_template.format(
        query=user_query,
        recipes=format_recipes(recipes),
        products=format_products(products),
    )

    print("Calling LLM...\n")
    answer = call_llm(system, user_message)
    print(answer)


if __name__ == "__main__":
    main()
