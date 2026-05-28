"""
Import recipe JSON files into MongoDB.

Usage:
    python import_recipes.py
"""

import json
from pathlib import Path
from pymongo import MongoClient

RECIPES_DIR = Path("data/recipes")
MONGO_URI = "mongodb://food_waste_mongo_user:food_waste_mongo_alex@food-waste-mongo:27017/"
MONGO_DB = "food_waste"
COLLECTION = "recipes"


def main():
    client = MongoClient(MONGO_URI)
    collection = client[MONGO_DB][COLLECTION]

    files = list(RECIPES_DIR.glob("*.json"))
    print(f"Found {len(files)} recipe files in {RECIPES_DIR}")

    ok = skipped = failed = 0

    for path in files:
        slug = path.stem
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if "error" in doc and "title" not in doc:
                skipped += 1
                continue
            doc["_id"] = slug
            collection.replace_one({"_id": slug}, doc, upsert=True)
            ok += 1
        except Exception as e:
            print(f"  FAIL {slug}: {e}")
            failed += 1

    print(f"Done. {ok} imported, {skipped} skipped (error files), {failed} failed.")
    print(f"Total documents in collection: {collection.count_documents({})}")
    client.close()


if __name__ == "__main__":
    main()