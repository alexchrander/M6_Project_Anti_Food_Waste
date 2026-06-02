from pymongo import MongoClient
import mysql.connector
import chromadb

# ── MongoDB ───────────────────────────────────────────────────────────────────
try:
    mongo = MongoClient("mongodb://food_waste_mongo_user:food_waste_mongo_alex@food-waste-mongo:27017/")
    db = mongo["food_waste"]
    print("MongoDB connected ✅")

    for name in db.list_collection_names():
        count = db[name].count_documents({})
        print(f"  {name}: {count} documents")
except Exception as e:
    print("MongoDB failed ❌", e)

# ── MySQL ─────────────────────────────────────────────────────────────────────
try:
    sql = mysql.connector.connect(
        host="food-waste-mysql",
        user="food_waste_mysql_user",
        password="food_waste_mysql_alex",
        database="food_waste_mysql"
    )
    print("\nMySQL connected ✅")
    cursor = sql.cursor()

    cursor.execute("SHOW TABLES;")
    tables = [row[0] for row in cursor.fetchall()]
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM `{table}`;")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count} rows")
except Exception as e:
    print("MySQL failed ❌", e)

# ── ChromaDB ──────────────────────────────────────────────────────────────────
try:
    chroma = chromadb.PersistentClient(path="data/chroma_db")
    print("\nChromaDB connected ✅")

    for col in chroma.list_collections():
        count = chroma.get_collection(col.name).count()
        print(f"  {col.name}: {count} embeddings")
except Exception as e:
    print("ChromaDB failed ❌", e)