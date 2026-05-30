"""
Build ChromaDB vector index from recipes stored in MongoDB.

Reads all recipes from MongoDB, embeds them using Gemini's embedding model
(task_type=RETRIEVAL_DOCUMENT), and stores the vectors in a local ChromaDB
collection for later retrieval.

Usage:
    python rag_pipeline/build_index.py
    python rag_pipeline/build_index.py --limit 5
"""

import argparse
import re
import sys
import time
from pathlib import Path

import chromadb
from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import embed_ingredients, embed_recipe

CHROMA_PATH = "data/chroma_db"
CHROMA_COLLECTION = "recipes"
INGREDIENT_COLLECTION = "recipe_ingredients"
REQUEST_DELAY = 0.5  # seconds between Gemini API calls


def clean_ingredients(ingredients: list[str]) -> list[str]:
    """Strip quantities, units and notes from raw ingredient strings."""
    cleaned = []
    for ing in ingredients:
        ing = ing.strip()
        if ing.startswith("*"):
            continue
        ing = re.sub(r'\(.*?\)', '', ing)  # strip parenthetical content
        ing = re.sub(                      # strip leading quantity + unit
            r'^[\d½¼⅓⅔¾,./\s]+ *(g|kg|dl|l|ml|stk|spsk|tsk|liter|cl|håndfuld|fed)\s*',
            '', ing, flags=re.IGNORECASE,
        )
        ing = re.sub(r'^[\d½¼⅓⅔¾]+\s+', '', ing)  # strip bare leading numbers
        ing = ing.strip()
        if ing:
            cleaned.append(ing)
    return cleaned


def build_embedding_text(doc: dict) -> str:
    """Concatenate the semantically useful fields into one string to embed."""
    parts = [
        doc.get("title", ""),
        doc.get("description", ""),
        ", ".join(doc.get("categories", [])),
        ", ".join(doc.get("keywords", [])),
        ", ".join(doc.get("ingredients", [])),
    ]
    return " ".join(p for p in parts if p)


def build_ingredient_index(mongo_collection, chroma_client, limit: int = 0) -> None:
    """Embed cleaned ingredient lists into the recipe_ingredients ChromaDB collection."""
    collection = chroma_client.get_or_create_collection(
        name=INGREDIENT_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(collection.get(include=[])["ids"])
    print(f"Already indexed: {len(existing_ids)} recipes in {INGREDIENT_COLLECTION}.")

    query = mongo_collection.find(
        {"_id": {"$nin": list(existing_ids)}, "ingredients": {"$exists": True, "$ne": []}},
        {"ingredients": 1, "title": 1},
    )
    if limit > 0:
        query = query.limit(limit)
    recipes = list(query)
    total = len(recipes)
    print(f"Found {total} new recipes to index.\n")

    ok = failed = 0
    for i, doc in enumerate(recipes, start=1):
        slug = doc["_id"]
        try:
            ingredients = clean_ingredients(doc.get("ingredients", []))
            if not ingredients:
                print(f"[{i}/{total}] SKIP {slug} — no usable ingredients")
                continue
            vector = embed_ingredients(ingredients)
            collection.upsert(
                ids=[slug],
                embeddings=[vector],
                metadatas=[{"title": doc.get("title", "")}],
            )
            print(f"[{i}/{total}] OK  {doc.get('title') or slug}")
            ok += 1
        except Exception as exc:
            print(f"[{i}/{total}] FAIL {slug} — {exc}")
            failed += 1

        if i < total:
            time.sleep(REQUEST_DELAY)

    print(f"\nDone. {ok} indexed, {failed} failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB index from MongoDB recipes.")
    parser.add_argument("--limit", type=int, default=0, help="Number of recipes to index (0 = all)")
    args = parser.parse_args()

    mongo_client = MongoClient(
        "mongodb://food_waste_mongo_user:food_waste_mongo_alex@food-waste-mongo:27017/"
    )
    collection_mongo = mongo_client["food_waste"]["recipes"]

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection_chroma = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(collection_chroma.get(include=[])["ids"])
    print(f"Already indexed: {len(existing_ids)} recipes in ChromaDB.")

    query = collection_mongo.find({"_id": {"$nin": list(existing_ids)}})
    if args.limit > 0:
        query = query.limit(args.limit)
    recipes = list(query)
    total = len(recipes)
    print(f"Found {total} new recipes to index.\n")

    ok = 0
    failed = 0

    for i, doc in enumerate(recipes, start=1):
        slug = doc["_id"]
        try:
            text = build_embedding_text(doc)
            vector = embed_recipe(text)

            collection_chroma.upsert(
                ids=[slug],
                embeddings=[vector],
                metadatas=[{
                    "title": doc.get("title", ""),
                    "url": doc.get("url", ""),
                }],
            )
            print(f"[{i}/{total}] OK  {doc.get('title') or slug}")
            ok += 1
        except Exception as exc:
            print(f"[{i}/{total}] FAIL {slug} — {exc}")
            failed += 1

        if i < total:
            time.sleep(REQUEST_DELAY)

    mongo_client.close()
    print(f"\nDone. {ok} indexed, {failed} failed. ChromaDB at: {CHROMA_PATH}")

    print(f"\n--- Building {INGREDIENT_COLLECTION} ---")
    mongo_client = MongoClient(
        "mongodb://food_waste_mongo_user:food_waste_mongo_alex@food-waste-mongo:27017/"
    )
    collection_mongo = mongo_client["food_waste"]["recipes"]
    build_ingredient_index(collection_mongo, chroma_client, args.limit)
    mongo_client.close()


if __name__ == "__main__":
    main()
