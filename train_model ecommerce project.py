"""
==========================================================
  OLIST E-COMMERCE  —  train_model.py
  Predicts: payment_value (regression)
  Saves  : ecommerce_model.pkl + model_features.pkl
  Run    : python train_model.py
==========================================================
"""

import os, gc, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 0.  LOCATE FILES
# ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

def load(name):
    path = os.path.join(BASE, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"\n❌  Missing file: {path}\n   Put train_model.py next to your CSV files.\n")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(f"   ✅  {name:45s}  {df.shape}")
    return df

# ──────────────────────────────────────────────
# 1.  LOAD
# ──────────────────────────────────────────────
print("\n📂  Loading CSV files ...")
orders    = load("olist_orders_dataset.csv")
items     = load("olist_order_items_dataset.csv")
payments  = load("olist_order_payments_dataset.csv")
products  = load("olist_products_dataset.csv")
customers = load("olist_customers_dataset.csv")
reviews   = load("olist_order_reviews_dataset.csv")

# ──────────────────────────────────────────────
# 2.  AGGREGATE PAYMENTS  →  TARGET
# ──────────────────────────────────────────────
print("\n🔧  Engineering features ...")

pay = (payments.groupby("order_id")
       .agg(payment_value        = ("payment_value",        "sum"),
            payment_installments = ("payment_installments", "max"),
            num_payment_types    = ("payment_type",         "nunique"))
       .reset_index())

# ──────────────────────────────────────────────
# 3.  AGGREGATE ITEMS
# ──────────────────────────────────────────────
itm = (items.groupby("order_id")
       .agg(num_items           = ("order_item_id", "count"),
            total_price         = ("price",         "sum"),
            mean_price          = ("price",         "mean"),
            total_freight       = ("freight_value", "sum"),
            num_unique_products = ("product_id",    "nunique"),
            num_unique_sellers  = ("seller_id",     "nunique"))
       .reset_index())

# ──────────────────────────────────────────────
# 4.  AGGREGATE REVIEWS
# ──────────────────────────────────────────────
rev = (reviews.groupby("order_id")[["review_score"]]
       .mean().reset_index())

# ──────────────────────────────────────────────
# 5.  PRODUCT WEIGHT / SIZE
# ──────────────────────────────────────────────
pcols = ["product_id"] + [c for c in
         ["product_weight_g","product_length_cm","product_height_cm","product_width_cm"]
         if c in products.columns]
ip = items[["order_id","product_id"]].merge(products[pcols], on="product_id", how="left")
prod_agg = (ip.groupby("order_id")
            .agg(mean_weight = ("product_weight_g",  "mean"),
                 mean_length = ("product_length_cm", "mean"))
            .reset_index())
del ip; gc.collect()

# ──────────────────────────────────────────────
# 6.  ORDER DATE FEATURES  (no leakage)
# ──────────────────────────────────────────────
LEAK = ["order_approved_at","order_delivered_carrier_date",
        "order_delivered_customer_date","order_status"]
orders.drop(columns=[c for c in LEAK if c in orders.columns], inplace=True)

if "order_purchase_timestamp" in orders.columns:
    ts = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce")
    orders["purchase_year"]    = ts.dt.year.astype("Int16")
    orders["purchase_month"]   = ts.dt.month.astype("Int8")
    orders["purchase_day"]     = ts.dt.day.astype("Int8")
    orders["purchase_hour"]    = ts.dt.hour.astype("Int8")
    orders["purchase_weekday"] = ts.dt.weekday.astype("Int8")
    orders.drop(columns=["order_purchase_timestamp"], inplace=True)

orders.drop(columns=[c for c in ["order_estimated_delivery_date"]
                     if c in orders.columns], inplace=True)

# ──────────────────────────────────────────────
# 7.  CUSTOMER STATE  (encoded)
# ──────────────────────────────────────────────
cust = customers[["customer_id","customer_state"]].copy()
cust["customer_state"] = cust["customer_state"].astype("category").cat.codes

# ──────────────────────────────────────────────
# 8.  MERGE
# ──────────────────────────────────────────────
print("\n🔗  Merging all tables ...")
df = orders.copy()
df = df.merge(pay,      on="order_id",    how="inner")
df = df.merge(itm,      on="order_id",    how="left")
df = df.merge(rev,      on="order_id",    how="left")
df = df.merge(prod_agg, on="order_id",    how="left")
df = df.merge(cust,     on="customer_id", how="left")

df.drop(columns=[c for c in ["order_id","customer_id"] if c in df.columns], inplace=True)

del orders, items, payments, products, customers, reviews
del pay, itm, rev, prod_agg, cust
gc.collect()

print(f"   ✅  Merged shape : {df.shape}")

# ──────────────────────────────────────────────
# 9.  CLEAN
# ──────────────────────────────────────────────
TARGET = "payment_value"
df = df.dropna(subset=[TARGET])
df = df[df[TARGET] > 0]
df.fillna(df.median(numeric_only=True), inplace=True)

# drop any stray object columns
obj = df.select_dtypes(include="object").columns.tolist()
if obj:
    df.drop(columns=obj, inplace=True)

print(f"   ✅  Clean shape  : {df.shape}")

# ──────────────────────────────────────────────
# 10.  SAMPLE  (RAM safety)
# ──────────────────────────────────────────────
MAX = 80_000
if len(df) > MAX:
    print(f"\n⚠️   Sampling {MAX:,} / {len(df):,} rows to protect RAM ...")
    df = df.sample(n=MAX, random_state=42).reset_index(drop=True)

# ──────────────────────────────────────────────
# 11.  SPLIT
# ──────────────────────────────────────────────
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(np.float32)
del df; gc.collect()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\n✅  Train : {X_train.shape}   Test : {X_test.shape}")
print(f"   Features ({len(X.columns)}) : {list(X.columns)}\n")

# ──────────────────────────────────────────────
# 12.  TRAIN
# ──────────────────────────────────────────────
print("🚀  Training Random Forest ...  (2-4 min, please wait)\n")

model = RandomForestRegressor(
    n_estimators    = 100,
    max_depth       = 12,
    min_samples_leaf= 3,
    max_features    = "sqrt",
    random_state    = 42,
    n_jobs          = 1,
)
model.fit(X_train, y_train)
print("   ✅  Training done!\n")

# ──────────────────────────────────────────────
# 13.  EVALUATE
# ──────────────────────────────────────────────
y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
r2     = r2_score(y_test, y_pred)

print("📊  ══════════════════════════════════════")
print("    Model Performance on Test Set         ")
print("    ══════════════════════════════════════")
print(f"    MAE   →  R$ {mae:.2f}  (average error)")
print(f"    RMSE  →  R$ {rmse:.2f}")
print(f"    R²    →  {r2:.4f}   (1.0 = perfect)")
print("    ══════════════════════════════════════\n")

fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("🔍  Top 10 features:")
for f, v in fi.head(10).items():
    print(f"    {f:<30s}  {v:.4f}  {'█' * int(v*150)}")

# ──────────────────────────────────────────────
# 14.  SAVE
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE, "ecommerce_model.pkl")
FEAT_PATH  = os.path.join(BASE, "model_features.pkl")

joblib.dump(model,           MODEL_PATH)
joblib.dump(list(X.columns), FEAT_PATH)

print(f"\n✅  Model saved    →  {MODEL_PATH}")
print(f"✅  Features saved →  {FEAT_PATH}")
print("\n🎉  Done!  Now run:  streamlit run app.py\n")

