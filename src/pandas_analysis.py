"""pandas cross-check of Q2 and exploratory statistics of completed order_amount.
Run: python src/pandas_analysis.py"""
import sqlite3

import numpy as np
import pandas as pd

con = sqlite3.connect("data/instamart.db")
df = pd.read_sql_query(
    "SELECT o.*, c.city, p.category FROM orders o "
    "JOIN customers c USING(customer_id) JOIN products p USING(product_id)", con)
done = df[df["status"] == "completed"]

# (a) pivot table with margins must equal the SQL grand total
pv = pd.pivot_table(done, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)
print(pv.round(2).to_string())
sql_total = pd.read_sql_query("SELECT SUM(order_amount) AS t FROM orders WHERE status='completed'", con)["t"][0]
print(f"\npivot grand total {pv.loc['All', 'All']:,.2f} | SQL total {sql_total:,.2f} | equal: {np.isclose(pv.loc['All', 'All'], sql_total)}")

by_city = done.groupby("city")["order_amount"].agg(total="sum", mean="mean", count="count").round(2)
print("\nPer-city sum / mean / count of completed order_amount")
print(by_city.to_string())

# (b) distribution statistics
x = done["order_amount"]
q1, q2, q3 = x.quantile([.25, .5, .75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\nmean {x.mean():.2f} | median {x.median():.2f} | skew {x.skew():.2f}")
print(f"Q1 {q1:.2f} | Q2 {q2:.2f} | Q3 {q3:.2f} | IQR {iqr:.2f}")
print(f"fences: lower {lo:.2f}, upper {hi:.2f} | outliers below {int((x < lo).sum())}, above {int((x > hi).sum())}")

cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr = done[cols].corr()
print("\nCorrelation matrix (completed orders)")
print(corr.round(3).to_string())
pairs = corr.where(~np.eye(len(cols), dtype=bool)).abs().stack()
best = pairs.idxmax()
print(f"largest |r|: {best[0]} vs {best[1]} = {corr.loc[best]:.3f}")
