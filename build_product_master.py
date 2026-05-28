"""
Build and sync a product master list into the MySQL `products` table.

- New EANs are inserted.
- Existing EANs are updated if any descriptive column changed (and updated_at is set).
- last_seen and times_on_clearance are always updated.
- first_seen is never changed after initial insert.

Usage:
    python build_product_master.py
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import chromadb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fetch_prediction_pipeline.store_sql import get_connection, init_products_table
from ml_pipeline.build_features import engineer_category
from llm_pipeline.embeddings import embed_product

CHROMA_PATH = "data/chroma_db"
CHROMA_COLLECTION = "clearance_products"

DESCRIPTIVE_COLS = ["product_description", "product_image", "category_level1_da", "category_level2_da", "category_level3_da", "category_level4_da", "category_level1_en", "category_level2_en", "category_level3_en", "category_level4_en", "offer_stock_unit"]


def build_master() -> pd.DataFrame:
    conn = get_connection()
    cols = "unique_id, product_ean, product_description, product_image, product_category_da, product_category_en, offer_stock_unit, fetched_at, offer_start_time"
    current = pd.read_sql(f"SELECT {cols} FROM current", conn)
    history = pd.read_sql(f"SELECT {cols} FROM history", conn)
    conn.close()

    df = pd.concat([current, history], ignore_index=True)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    df["offer_start_time"] = pd.to_datetime(df["offer_start_time"])

    seen = df.groupby("product_ean")["offer_start_time"].agg(first_seen="min", last_seen="max").reset_index()

    times = (
        df.dropna(subset=["unique_id"])
        .groupby("product_ean")["unique_id"]
        .nunique()
        .reset_index()
        .rename(columns={"unique_id": "times_on_clearance"})
    )

    df_sorted = df.sort_values("fetched_at", ascending=False)

    def most_recent_non_null(series):
        non_null = series.dropna()
        return non_null.iloc[0] if len(non_null) > 0 else None

    agg = (
        df_sorted.groupby("product_ean")[
            ["product_description", "product_image", "product_category_da", "product_category_en", "offer_stock_unit"]
        ]
        .agg(most_recent_non_null)
        .reset_index()
    )

    agg = engineer_category(agg)

    master = agg.merge(seen, on="product_ean").merge(times, on="product_ean")
    return master[[
        "product_ean", "product_description", "product_image",
        "category_level1_da", "category_level2_da", "category_level3_da", "category_level4_da",
        "category_level1_en", "category_level2_en", "category_level3_en", "category_level4_en",
        "offer_stock_unit", "first_seen", "last_seen", "times_on_clearance",
    ]]


def sync_to_mysql(master: pd.DataFrame) -> None:
    conn = get_connection()
    init_products_table(conn)
    cursor = conn.cursor()

    cursor.execute("SELECT product_ean, product_description, product_image, category_level1_da, category_level2_da, category_level3_da, category_level4_da, category_level1_en, category_level2_en, category_level3_en, category_level4_en, offer_stock_unit FROM products")
    existing = {row[0]: row[1:] for row in cursor.fetchall()}

    now = datetime.utcnow()
    inserted = updated = skipped = 0

    for _, row in master.iterrows():
        ean = row["product_ean"]
        desc_values = tuple(row[c] for c in DESCRIPTIVE_COLS)

        if ean not in existing:
            cursor.execute("""
                INSERT INTO products (product_ean, product_description, product_image,
                    category_level1_da, category_level2_da, category_level3_da, category_level4_da,
                    category_level1_en, category_level2_en, category_level3_en, category_level4_en,
                    offer_stock_unit, first_seen, last_seen, times_on_clearance, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (ean, *desc_values, row["first_seen"], row["last_seen"], int(row["times_on_clearance"]), now))
            inserted += 1
        else:
            existing_desc = existing[ean]
            desc_changed = any(
                str(a or "") != str(b or "")
                for a, b in zip(desc_values, existing_desc)
            )

            if desc_changed:
                cursor.execute("""
                    UPDATE products
                    SET product_description = %s, product_image = %s,
                        category_level1_da = %s, category_level2_da = %s,
                        category_level3_da = %s, category_level4_da = %s,
                        category_level1_en = %s, category_level2_en = %s,
                        category_level3_en = %s, category_level4_en = %s,
                        offer_stock_unit = %s, last_seen = %s,
                        times_on_clearance = %s, updated_at = %s
                    WHERE product_ean = %s
                """, (*desc_values, row["last_seen"], int(row["times_on_clearance"]), now, ean))
                updated += 1
            else:
                cursor.execute("""
                    UPDATE products
                    SET last_seen = %s, times_on_clearance = %s
                    WHERE product_ean = %s
                """, (row["last_seen"], int(row["times_on_clearance"]), ean))
                skipped += 1

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Inserted: {inserted}  |  Updated (desc changed): {updated}  |  Refreshed (stats only): {skipped}")


def sync_to_chroma(master: pd.DataFrame, reset: bool = False) -> None:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    if reset:
        try:
            chroma_client.delete_collection(CHROMA_COLLECTION)
            print(f"Deleted existing '{CHROMA_COLLECTION}' collection.")
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(collection.get(include=[])["ids"])
    to_embed = master[~master["product_ean"].astype(str).isin(existing_ids)]
    total = len(to_embed)
    print(f"Already in ChromaDB: {len(existing_ids)}  |  To embed: {total}")

    ok = failed = 0
    for i, (_, row) in enumerate(to_embed.iterrows(), start=1):
        ean = str(row["product_ean"])
        try:
            vector = embed_product(
                category1=row["category_level1_da"] or "",
                category2=row["category_level2_da"] or "",
                category3=row["category_level3_da"] or "",
                category4=row["category_level4_da"] or "",
                description=row["product_description"] or "",
            )
            collection.upsert(
                ids=[ean],
                embeddings=[vector],
                metadatas=[{
                    "category_level1_da": row["category_level1_da"] or "",
                    "category_level2_da": row["category_level2_da"] or "",
                    "category_level3_da": row["category_level3_da"] or "",
                    "category_level4_da": row["category_level4_da"] or "",
                }],
            )
            print(f"[{i}/{total}] OK  {ean} — {row['product_description']}")
            ok += 1
        except Exception as exc:
            print(f"[{i}/{total}] FAIL {ean} — {exc}")
            failed += 1

    print(f"ChromaDB sync done. {ok} embedded, {failed} failed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset-chroma", action="store_true", help="Delete and rebuild the clearance_products ChromaDB collection from scratch")
    args = parser.parse_args()

    print("Building product master list...")
    master = build_master()
    print(f"Unique products: {len(master)}")

    print("Syncing to MySQL products table...")
    sync_to_mysql(master)

    print("Syncing embeddings to ChromaDB...")
    sync_to_chroma(master, reset=args.reset_chroma)
    print("Done.")


if __name__ == "__main__":
    main()