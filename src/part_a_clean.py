"""Part A: audit raw exports, clean them with documented rules, store CSV + SQLite.
Run from the repository root:  python src/part_a_clean.py"""
import os
import sqlite3

import pandas as pd

RAW, CLEAN, DB = "data/raw", "data/clean", "data/instamart.db"
os.makedirs(CLEAN, exist_ok=True)


def missing_table(df):
    n = df.isna().sum()
    return pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": n, "missing_pct": (n / len(df) * 100).round(2)})


# ------------------------------------------------------------------ AUDIT
print("=" * 70, "\nAUDIT OF RAW DATA\n" + "=" * 70)
for name in ["customers", "products", "orders"]:
    df = pd.read_csv(f"{RAW}/{name}.csv")          # default NA handling: blanks and "N/A" read as missing
    print(f"\n--- {name}.csv  shape={df.shape}")
    print(missing_table(df).to_string())

# string-level reads so that nothing is silently converted
o_s = pd.read_csv(f"{RAW}/orders.csv", dtype=str, keep_default_na=False)
c_s = pd.read_csv(f"{RAW}/customers.csv", dtype=str, keep_default_na=False)
amt_num = pd.to_numeric(o_s["order_amount"], errors="coerce")
dm_num = pd.to_numeric(o_s["delivery_minutes"], errors="coerce")
age_num = pd.to_numeric(c_s["age"], errors="coerce")
audit = {
    "raw orders.csv rows": len(o_s),
    "exact duplicate rows": int(o_s.duplicated().sum()),
    "rows with status pending": int((o_s["status"] == "pending").sum()),
    "distinct order_id": int(o_s["order_id"].nunique()),
    "non-numeric order_amount": int(amt_num.isna().sum()),
    "negative delivery_minutes": int((dm_num < 0).sum()),
    "blank delivery_minutes": int((o_s["delivery_minutes"] == "").sum()),
    "blank age": int((c_s["age"] == "").sum()),
    "age outside 10-100": int(((age_num < 10) | (age_num > 100)).sum()),
    "distinct raw city spellings": int(c_s["city"].nunique()),
    "distinct non-blank raw gender spellings": int(c_s.loc[c_s["gender"] != "", "gender"].nunique()),
    "blank gender": int((c_s["gender"] == "").sum()),
}
print("\n--- Audit figures")
for k, v in audit.items():
    print(f"{k:45s}{v:>8,}")

# ------------------------------------------------------------------ CLEAN CUSTOMERS
print("\n" + "=" * 70, "\nCLEANING CUSTOMERS\n" + "=" * 70)
cust = pd.read_csv(f"{RAW}/customers.csv", dtype=str, keep_default_na=False)
before = cust["city"].copy()
cust["city"] = cust["city"].str.strip().str.title()                       # (a)
print(f"(a) city values changed by strip+title: {(before != cust['city']).sum()}; distinct now: {cust['city'].nunique()}")
gmap = {"m": "male", "male": "male", "f": "female", "female": "female", "": "unknown"}   # (b)
cust["gender"] = cust["gender"].str.strip().str.lower().map(gmap)
assert cust["gender"].notna().all()
age = pd.to_numeric(cust["age"], errors="coerce")                          # (c)
bad = (age < 10) | (age > 100)
age = age.mask(bad)
print(f"(c) impossible ages set to missing: {int(bad.sum())}; missing ages before fill: {int(age.isna().sum())}")
cust["age"] = age.fillna(age.groupby(cust["city"]).transform("median"))
cust["signup_date"] = pd.to_datetime(cust["signup_date"])                  # (d)
print("cities:", cust["city"].value_counts().to_dict())
print("genders:", cust["gender"].value_counts().to_dict())
print("missing ages:", int(cust["age"].isna().sum()), "| age range:", cust["age"].min(), "-", cust["age"].max())

# ------------------------------------------------------------------ CLEAN ORDERS
print("\n" + "=" * 70, "\nCLEANING ORDERS\n" + "=" * 70)
prod = pd.read_csv(f"{RAW}/products.csv")
orders = pd.read_csv(f"{RAW}/orders.csv", dtype=str, keep_default_na=False)
n0 = len(orders)
orders["order_ts"] = pd.to_datetime(orders["order_ts"])                    # rule 1
print(f"rule 1: order_ts converted to datetime ({n0:,} rows)")

n = len(orders)
orders = orders.drop_duplicates()                                          # rule 2
print(f"rule 2: exact duplicate rows removed: {n - len(orders):,} (rows now {len(orders):,})")

n = len(orders)                                                            # rule 3
orders = orders.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last").reset_index(drop=True)
print(f"rule 3: older records of repeated order_id removed (keep latest order_ts): {n - len(orders):,} (rows now {len(orders):,})")

orders["quantity"] = orders["quantity"].astype(int)
orders["discount_pct"] = orders["discount_pct"].astype(int)
orders = orders.merge(prod[["product_id", "category", "price_inr"]], on="product_id", how="left")
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")        # rule 4
miss = orders["order_amount"].isna()
rebuilt = (orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
orders.loc[miss, "order_amount"] = rebuilt[miss]
print(f"rule 4: unparseable order_amount values refilled from product price: {int(miss.sum())}")

dm = pd.to_numeric(orders["delivery_minutes"], errors="coerce")            # rule 5
neg = dm < 0
dm = dm.mask(neg)
orders["delivery_minutes"] = dm
done = orders["status"] == "completed"
med = orders.loc[done].groupby("category")["delivery_minutes"].transform("median")
blank_done = int(orders.loc[done, "delivery_minutes"].isna().sum())
orders.loc[done, "delivery_minutes"] = orders.loc[done, "delivery_minutes"].fillna(med)
print(f"rule 5: negative delivery_minutes set to missing: {int(neg.sum())}")
print(f"        completed orders with missing delivery filled by category median: {blank_done:,}")
print("        cancelled orders left missing on purpose:", int((orders['status'] == 'cancelled').sum()))
orders = orders.drop(columns=["category", "price_inr"])

print("\nrows:", len(orders), "| status counts:", orders["status"].value_counts().to_dict())
print("missing order_amount:", int(orders["order_amount"].isna().sum()))
print("missing delivery (completed):", int(orders.loc[orders["status"] == "completed", "delivery_minutes"].isna().sum()),
      "| missing delivery (cancelled):", int(orders.loc[orders["status"] == "cancelled", "delivery_minutes"].isna().sum()))
print(f"completed revenue: INR {orders.loc[orders['status'] == 'completed', 'order_amount'].sum():,.2f}")

# ------------------------------------------------------------------ STORE
orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
cust_out = cust.copy()
cust_out["signup_date"] = cust_out["signup_date"].dt.strftime("%Y-%m-%d")
cust_out.to_csv(f"{CLEAN}/customers.csv", index=False)
prod.to_csv(f"{CLEAN}/products.csv", index=False)
orders.to_csv(f"{CLEAN}/orders.csv", index=False)

if os.path.exists(DB):
    os.remove(DB)
with sqlite3.connect(DB) as con:
    cust_out.to_sql("customers", con, index=False)
    prod.to_sql("products", con, index=False)
    orders.to_sql("orders", con, index=False)
print(f"\nstored: {CLEAN}/*.csv and {DB} (tables customers, products, orders)")
