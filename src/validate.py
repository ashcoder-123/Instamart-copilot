"""Validate Part A reference checkpoints.  Run: python src/validate.py"""
import sqlite3

import pandas as pd

con = sqlite3.connect("data/instamart.db")
cust = pd.read_sql_query("SELECT * FROM customers", con)
orders = pd.read_sql_query("SELECT * FROM orders", con)
prod = pd.read_sql_query("SELECT * FROM products", con)
done = orders[orders["status"] == "completed"]
canc = orders[orders["status"] == "cancelled"]

checks = [
    ("orders rows == 12309", len(orders) == 12309),
    ("one row per order_id", orders["order_id"].is_unique),
    ("11110 completed", len(done) == 11110),
    ("1199 cancelled", len(canc) == 1199),
    ("no pending rows", (orders["status"] == "pending").sum() == 0),
    ("0 missing order_amount", orders["order_amount"].isna().sum() == 0),
    ("0 missing delivery among completed", done["delivery_minutes"].isna().sum() == 0),
    ("1199 missing delivery among cancelled", canc["delivery_minutes"].isna().sum() == 1199),
    ("completed revenue == 2998668.80", round(done["order_amount"].sum(), 2) == 2998668.80),
    ("six cities", cust["city"].nunique() == 6),
    ("city counts", cust["city"].value_counts().to_dict() ==
     {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}),
    ("gender counts", cust["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}),
    ("0 missing ages, all 18-60", cust["age"].isna().sum() == 0 and cust["age"].between(18, 60).all()),
    ("1200 customers, 40 products", len(cust) == 1200 and len(prod) == 40),
]
ok = True
for name, passed in checks:
    print(("PASS  " if passed else "FAIL  ") + name)
    ok &= bool(passed)
assert ok, "some checkpoints failed"
print("ALL CHECKPOINTS PASSED")
