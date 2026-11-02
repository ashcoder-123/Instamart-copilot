"""Part B (tasks 1-3): RFM table, K-means, segment profile, PCA, outputs/customer_segments.csv.
Run from the repository root:  python src/part_b_segments.py"""
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4

# ---------------------------------------------------------------- 1. RFM
orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
done = orders[orders["status"] == "completed"]
rfm = done.groupby("customer_id").agg(last_ts=("order_ts", "max"),
                                      frequency=("order_id", "nunique"),
                                      monetary=("order_amount", "sum"))
rfm["recency"] = (SNAPSHOT - rfm["last_ts"]).dt.days
rfm = rfm[["recency", "frequency", "monetary"]]
rfm["monetary"] = rfm["monetary"].round(2)
print("RFM customers:", len(rfm))
print(f"recency  median {rfm['recency'].median():.0f}  min {rfm['recency'].min()}  max {rfm['recency'].max()}")
print(f"frequency mean {rfm['frequency'].mean():.2f}  max {rfm['frequency'].max()}")
print(f"monetary  mean {rfm['monetary'].mean():.2f}  max {rfm['monetary'].max():.2f}")

# ---------------------------------------------------------------- 2. K-means for K = 2..8
scaler = StandardScaler()
X = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append({"K": k, "inertia": round(km.inertia_, 1), "silhouette": round(silhouette_score(X, km.labels_), 3)})
ktab = pd.DataFrame(rows)
print("\nK-means table")
print(ktab.to_string(index=False))

# ---------------------------------------------------------------- 3. final model, profile, names
km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("recency", "size"), mean_recency=("recency", "mean"),
                                     mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean")).round(2)

# Names are derived from the profile, never from the arbitrary cluster number.
remaining = set(prof.index)
premium_id = prof["mean_monetary"].idxmax(); remaining.discard(premium_id)
loyal_id = prof.loc[list(remaining), "mean_frequency"].idxmax(); remaining.discard(loyal_id)
risk_id = prof.loc[list(remaining), "mean_recency"].idxmax(); remaining.discard(risk_id)
(other_id,) = remaining
names = {premium_id: "Premium Big Spenders", loyal_id: "Loyal Regulars",
         risk_id: "At-Risk Fading", other_id: "Occasional Shoppers"}
actions = {"Premium Big Spenders": "VIP perks, early access to electronics launches, priority support",
           "Loyal Regulars": "Protect the habit: free-delivery thresholds, referral and streak rewards",
           "At-Risk Fading": "Win-back coupon and reminder before they lapse",
           "Occasional Shoppers": "Bundle offers and category nudges to raise order frequency"}
prof["segment_name"] = prof.index.map(names)
prof["action"] = prof["segment_name"].map(actions)
prof["is_premium"] = prof.index == premium_id
print("\nSegment profile (cluster numbers are only labels with no ordering)")
print(prof.to_string())

# ---------------------------------------------------------------- PCA
pca = PCA(n_components=2).fit(X)
evr = pca.explained_variance_ratio_
print(f"\nPCA explained_variance_ratio_: {evr[0]:.3f}, {evr[1]:.3f}, total {evr.sum():.3f}")

# ---------------------------------------------------------------- output file
out = rfm.reset_index()
out["segment_name"] = out["segment_id"].map(names)
out["is_premium"] = out["segment_id"] == premium_id
out[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]] \
    .to_csv("outputs/customer_segments.csv", index=False)
print("\nwrote outputs/customer_segments.csv:", len(out), "rows | premium customers:", int(out["is_premium"].sum()))
