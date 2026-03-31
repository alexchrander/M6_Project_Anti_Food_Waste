import sqlite3
conn = sqlite3.connect("data/food_waste.db")
cursor = conn.cursor()
cursor.execute("ALTER TABLE history RENAME COLUMN product_customer_flow TO store_customer_flow")
cursor.execute("ALTER TABLE current RENAME COLUMN product_customer_flow TO store_customer_flow")
conn.commit()
conn.close()
print("Done!")