"""Part B (tasks 4-8): lapse problem, leakage-free preprocessing, tuning, balancing, persistence, retention list.
Run after part_b_segments.py:  python src/part_b_lapse.py"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

CUTOFF = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2025-10-01")

# ---------------------------------------------------------------- 4. cohort, label, features
orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
cust = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"]).set_index("customer_id")
prod = pd.read_csv("data/clean/products.csv")
orders = orders.merge(prod[["product_id", "category"]], on="product_id", how="left")

before = orders[orders["order_ts"] < CUTOFF]            # ONLY these rows feed the features
before_done = before[before["status"] == "completed"]
sept_done = orders[(orders["status"] == "completed") & (orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < END)]

cohort_ids = before_done["customer_id"].unique()
y = pd.Series(1, index=cohort_ids, name="lapsed")
y[:] = (~pd.Index(cohort_ids).isin(sept_done["customer_id"].unique())).astype(int)
y.index.name = "customer_id"

g = before_done.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUTOFF - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount_pct": g["discount_pct"].mean(),
    "avg_delivery_minutes": g["delivery_minutes"].mean(),
    "fav_category": g["category"].agg(lambda s: s.mode().iloc[0]),
})
final_status = before[before["status"].isin(["completed", "cancelled"])]
feat["cancel_rate"] = (final_status.groupby("customer_id")["status"]
                       .apply(lambda s: (s == "cancelled").mean()).reindex(feat.index).fillna(0.0))
feat = feat.join(cust[["city", "age", "gender"]])
feat["tenure_days"] = (CUTOFF - cust.loc[feat.index, "signup_date"]).dt.days
NUM = ["recency", "frequency", "monetary", "avg_discount_pct", "cancel_rate", "avg_delivery_minutes", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X = feat[NUM + CAT]
assert X.isna().sum().sum() == 0, "feature matrix has missing values"
print(f"cohort: {len(X)} customers | lapsed: {int(y.sum())} ({y.mean() * 100:.2f}%)")
assert before["order_ts"].max() < CUTOFF

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"train {len(X_train)} | test {len(X_test)}")
base_pred = np.zeros(len(y_test), dtype=int)
print(f"baseline (always 'not lapsed'): accuracy {accuracy_score(y_test, base_pred) * 100:.1f}% | "
      f"lapsed-class F1 {f1_score(y_test, base_pred, zero_division=0):.3f}")


# ---------------------------------------------------------------- 5. preprocessing inside a Pipeline
def make_pipe(model):
    prep = ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])
    return Pipeline([("prep", prep), ("clf", model)])


def report(name, pipe, Xt, yt):
    pred = pipe.predict(Xt)
    cm = confusion_matrix(yt, pred)
    p, r, f = (precision_score(yt, pred, zero_division=0), recall_score(yt, pred, zero_division=0),
               f1_score(yt, pred, zero_division=0))
    print(f"{name}: confusion [[TN FP],[FN TP]] = {cm.tolist()} | precision {p:.3f} recall {r:.3f} F1 {f:.3f}")
    return {"precision": p, "recall": r, "f1": f}


probe = make_pipe(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
learned = probe.named_steps["prep"].named_transformers_["num"].mean_[NUM.index("frequency")]
print(f"\nscaler mean of frequency (train rows only): {learned:.4f} | full-cohort mean: {X['frequency'].mean():.4f}")
print("They differ because the scaler saw only the 924 training rows, never the test rows.")
print("missing values in feature columns:", int(X.isna().sum().sum()))

# ---------------------------------------------------------------- 6. tuning
lr_gs = GridSearchCV(make_pipe(LogisticRegression(max_iter=1000)), {"clf__C": [0.01, 0.1, 1, 10, 100]},
                     cv=5, scoring="f1").fit(X_train, y_train)
rf_gs = GridSearchCV(make_pipe(RandomForestClassifier(n_estimators=100, random_state=42)),
                     {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                     cv=5, scoring="f1").fit(X_train, y_train)
print(f"\nLogReg best params {lr_gs.best_params_} | best CV F1 {lr_gs.best_score_:.4f}")
print(f"RandForest best params {rf_gs.best_params_} | best CV F1 {rf_gs.best_score_:.4f}")
print("\nTest-set evaluation (lapsed = positive class)")
m_lr = report("LogReg ", lr_gs.best_estimator_, X_test, y_test)
m_rf = report("RandFor ", rf_gs.best_estimator_, X_test, y_test)

# model choice made on cross-validated F1 (not on the test set)
if rf_gs.best_score_ >= lr_gs.best_score_:
    chosen_name, chosen = "RandomForest", rf_gs.best_estimator_
else:
    chosen_name, chosen = "LogisticRegression", lr_gs.best_estimator_
print("selected model:", chosen_name)

# ---------------------------------------------------------------- 7. class-balance experiment (train only)
lapsed_rows = X_train[y_train == 1]
n_add = int((y_train == 0).sum() - (y_train == 1).sum())
extra = resample(lapsed_rows, replace=True, n_samples=n_add, random_state=42)
X_bal = pd.concat([X_train, extra])
y_bal = pd.concat([y_train, pd.Series(1, index=extra.index)])
print(f"\nbalanced training counts: {y_bal.value_counts().to_dict()}")
best_lr = clone(lr_gs.best_estimator_).fit(X_bal, y_bal)
print("Test metrics, original un-balanced test set")
m_unb = report("LogReg un-balanced", lr_gs.best_estimator_, X_test, y_test)
m_bal = report("LogReg balanced   ", best_lr, X_test, y_test)

# ---------------------------------------------------------------- 8. interpretation, persistence, retention list
names = chosen.named_steps["prep"].get_feature_names_out()
rf_names = rf_gs.best_estimator_.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf_gs.best_estimator_.named_steps["clf"].feature_importances_, index=rf_names).sort_values(ascending=False)
print("\nTop 5 random-forest feature importances")
print(imp.head(5).round(4).to_string())
lr_names = lr_gs.best_estimator_.named_steps["prep"].get_feature_names_out()
coef = pd.Series(lr_gs.best_estimator_.named_steps["clf"].coef_[0], index=lr_names)
print("\nLogistic regression coefficients (standardised numeric features)")
for f in ["num__recency", "num__frequency", "num__monetary", "num__avg_discount_pct", "num__cancel_rate"]:
    print(f"{f:28s}{coef[f]:+.4f}")

joblib.dump(chosen, "outputs/lapse_model.joblib")
loaded = joblib.load("outputs/lapse_model.joblib")
same = bool((loaded.predict(X_test) == chosen.predict(X_test)).all()) and \
    bool(np.allclose(loaded.predict_proba(X_test), chosen.predict_proba(X_test)))
print("\nreloaded model reproduces test predictions exactly:", same)
assert same

seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
premium_ids = set(seg.index[seg["is_premium"]])
test_prem = X_test[X_test.index.isin(premium_ids)]
proba = pd.Series(chosen.predict_proba(test_prem)[:, 1], index=test_prem.index)
top = proba.sort_values(ascending=False).head(10)
ret = pd.DataFrame({"customer_id": top.index, "lapse_probability": top.values.round(4),
                    "actually_lapsed": y_test.loc[top.index].values})
ret.to_csv("outputs/retention_list.csv", index=False)
print(f"\nPremium customers in test set: {len(test_prem)} | retention list rows: {len(ret)} | "
      f"truly lapsed: {int(ret['actually_lapsed'].sum())}")
print(ret.to_string(index=False))
print("NOTE: the Premium flag uses whole-window RFM (includes September), so this list is illustrative, not a leakage-free evaluation.")
