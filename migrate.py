# migrate2.py — run once, then delete
import sqlite3

conn = sqlite3.connect("data/food_waste.db")
cursor = conn.cursor()

# SQLite doesn't support DROP COLUMN directly in older versions
# so we recreate both tables without store_customer_flow
# and with the two new columns instead

migrations = [
    # Add two new columns to history
    "ALTER TABLE history ADD COLUMN store_customer_flow_today    TEXT",
    "ALTER TABLE history ADD COLUMN store_customer_flow_tomorrow TEXT",
    # Add two new columns to current
    "ALTER TABLE current ADD COLUMN store_customer_flow_today    TEXT",
    "ALTER TABLE current ADD COLUMN store_customer_flow_tomorrow TEXT",
]

for sql in migrations:
    try:
        cursor.execute(sql)
        print(f"✓ {sql}")
    except Exception as e:
        print(f"✗ Skipped: {e}")

# Drop the old column by setting all values to NULL
# SQLite ALTER TABLE doesn't support DROP COLUMN in older versions
# so we just zero it out — we'll remove it from all future inserts
cursor.execute("UPDATE history SET store_customer_flow = NULL")
cursor.execute("UPDATE current SET store_customer_flow = NULL")

conn.commit()
conn.close()
print("\nMigration complete!")