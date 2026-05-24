"""
Build ChromaDB vector index from recipes stored in MongoDB.

Reads all recipes from MongoDB, embeds them using Gemini's embedding model
(task_type=RETRIEVAL_DOCUMENT), and stores the vectors in a local ChromaDB
collection for later retrieval.

Usage:
    python llm_dev/build_index.py
    python llm_dev/build_index.py --limit 5
"""

import argparse
import sys
import time
from pathlib import Path

import chromadb
from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import embed_recipe

CHROMA_PATH = "data/chroma_db"
CHROMA_COLLECTION = "recipes"
REQUEST_DELAY = 0.5  # seconds between Gemini API calls


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

    query = collection_mongo.find()
    if args.limit > 0:
        query = query.limit(args.limit)
    recipes = list(query)
    total = len(recipes)
    print(f"Found {total} recipes in MongoDB.\n")

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


if __name__ == "__main__":
    main()
