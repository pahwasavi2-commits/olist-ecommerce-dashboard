"""
==========================================================
  OLIST E-COMMERCE  —  app.py
  Run: streamlit run app.py
==========================================================
"""

import os, gc, warnings, sqlite3
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "Olist E-Commerce Intelligence",
    page_icon  = "🛒",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ══════════════════════════════════════════════════════════
#  STYLING
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
:root{
  --bg:#0d0f14; --surface:#151820; --card:#1b1f2a; --border:#272c3a;
  --accent:#f97316; --blue:#3b82f6; --green:#10b981; --red:#ef4444;
  --text:#e2e8f0; --muted:#64748b;
  --mono:'Space Mono',monospace; --sans:'Syne',sans-serif;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:var(--sans)!important;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stDecoration"]{display:none;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
[data-testid="stMetric"]{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:20px 24px!important;}
[data-testid="stMetric"]:hover{border-color:var(--accent)!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-family:var(--mono)!important;font-size:11px!important;text-transform:uppercase;letter-spacing:1px;}
[data-testid="stMetricValue"]{color:var(--text)!important;font-family:var(--sans)!important;font-size:2rem!important;font-weight:800!important;}
.stButton>button{background:var(--accent)!important;color:#fff!important;border:none!important;border-radius:8px!important;font-family:var(--mono)!important;font-size:13px!important;font-weight:700!important;letter-spacing:1px!important;padding:12px 32px!important;}
.stButton>button:hover{opacity:.85!important;transform:translateY(-1px)!important;}
[data-baseweb="tab-list"]{background:var(--surface)!important;border-radius:10px;padding:4px;gap:4px;}
[data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;border-radius:8px!important;font-family:var(--mono)!important;font-size:12px!important;}
[aria-selected="true"]{background:var(--accent)!important;color:#fff!important;}
.sec{font-family:var(--sans);font-size:.95rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:2px;margin:2rem 0 1rem;padding-bottom:8px;border-bottom:1px solid var(--border);}
.hero{background:linear-gradient(135deg,#1b1f2a,#0d1117);border:1px solid var(--border);border-radius:16px;padding:36px 40px;margin-bottom:2rem;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-60px;right:-60px;width:220px;height:220px;background:radial-gradient(circle,rgba(249,115,22,.15),transparent 70%);border-radius:50%;}
.hero h1{font-family:var(--sans);font-size:2.2rem;font-weight:800;color:var(--text);margin:0 0 6px;}
.hero p{color:var(--muted);font-family:var(--mono);font-size:13px;margin:0;}
.hero .tag{display:inline-block;background:rgba(249,115,22,.15);color:var(--accent);border:1px solid rgba(249,115,22,.3);border-radius:20px;font-family:var(--mono);font-size:11px;padding:3px 12px;margin-bottom:12px;}
.pcard{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:24px;text-align:center;}
.pcard .big{font-size:2.5rem;font-weight:800;font-family:var(--sans);}
.pcard .lbl{font-family:var(--mono);font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════
BASE = os.path.dirname(os.path.abspath(__file__))

def csv(name):
    p = os.path.join(BASE, name)
    if not os.path.exists(p):
        st.error(f"❌ File not found: **{name}** — make sure it's in the same folder as app.py")
        st.stop()
    return p

def dark(fig, ax, xgrid=False, ygrid=True):
    fig.patch.set_facecolor("#1b1f2a")
    ax.set_facecolor("#1b1f2a")
    ax.tick_params(colors="#64748b", labelsize=8)
    for s in ax.spines.values(): s.set_visible(False)
    if ygrid: ax.grid(axis="y", color="#272c3a", linewidth=0.6)
    if xgrid: ax.grid(axis="x", color="#272c3a", linewidth=0.6)

def sec(label):
    st.markdown(f'<div class="sec">{label}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════
@st.cache_data(show_spinner="⏳ Loading Olist dataset ...")
def load_data():
    orders    = pd.read_csv(csv("olist_orders_dataset.csv"),         low_memory=False)
    items     = pd.read_csv(csv("olist_order_items_dataset.csv"),    low_memory=False)
    payments  = pd.read_csv(csv("olist_order_payments_dataset.csv"), low_memory=False)
    customers = pd.read_csv(csv("olist_customers_dataset.csv"),      low_memory=False)
    reviews   = pd.read_csv(csv("olist_order_reviews_dataset.csv"),  low_memory=False)

    for d in [orders, items, payments, customers, reviews]:
        d.columns = [c.strip().lower().replace(" ","_") for c in d.columns]

    # dates
    for col in ["order_purchase_timestamp","order_delivered_customer_date","order_estimated_delivery_date"]:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")

    orders["delivery_delay_days"] = (
        orders["order_delivered_customer_date"] -
        orders["order_estimated_delivery_date"]
    ).dt.days

    orders["month"]   = orders["order_purchase_timestamp"].dt.month
    orders["hour"]    = orders["order_purchase_timestamp"].dt.hour
    orders["year"]    = orders["order_purchase_timestamp"].dt.year
    orders["weekday"] = orders["order_purchase_timestamp"].dt.day_name()

    # merges
    pay = payments.groupby("order_id")["payment_value"].sum().reset_index()
    orders = orders.merge(pay, on="order_id", how="left")
    orders = orders.merge(customers[["customer_id","customer_state"]], on="customer_id", how="left")
    rev = reviews.groupby("order_id")["review_score"].mean().reset_index()
    orders = orders.merge(rev, on="order_id", how="left")
    itm = (items.groupby("order_id")
           .agg(num_items=("order_item_id","count"),
                total_freight=("freight_value","sum"),
                mean_price=("price","mean"))
           .reset_index())
    orders = orders.merge(itm, on="order_id", how="left")

    return orders

df = load_data()

# ══════════════════════════════════════════════════════════
#  LOAD MODEL  (safe — no crash if missing)
# ══════════════════════════════════════════════════════════
MODEL_PATH = os.path.join(BASE, "ecommerce_model.pkl")
FEAT_PATH  = os.path.join(BASE, "model_features.pkl")
model = feat = None
if os.path.exists(MODEL_PATH) and os.path.exists(FEAT_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        feat  = joblib.load(FEAT_PATH)
    except Exception as e:
        st.warning(f"⚠️ Model load error: {e}")

# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🛒 Olist Intelligence")
    st.markdown("---")
    s_status  = st.multiselect("Order Status",
                    sorted(df["order_status"].dropna().unique()), 
                    default=df["order_status"].dropna().unique().tolist())
    s_year    = st.multiselect("Year",
                    sorted(df["year"].dropna().unique().astype(int).tolist()),
                    default=sorted(df["year"].dropna().unique().astype(int).tolist()))
    s_month   = st.slider("Month Range", 1, 12, (1,12))
    s_state   = st.selectbox("Customer State",
                    ["All"] + sorted(df["customer_state"].dropna().unique().tolist()))
    st.markdown("---")
    st.caption("Olist Brazilian E-Commerce · Public Dataset")

fdf = df[
    df["order_status"].isin(s_status) &
    df["year"].isin(s_year) &
    df["month"].between(s_month[0], s_month[1])
].copy()
if s_state != "All":
    fdf = fdf[fdf["customer_state"] == s_state]

# ══════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="tag">LIVE DASHBOARD</div>
  <h1>🛒 Olist E-Commerce Intelligence</h1>
  <p>Analytics · Delivery Insights · Revenue · AI Prediction · SQL Explorer</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════
t1, t2, t3, t4, t5, t6 = st.tabs([
    "📊 Overview", "🚚 Delivery", "💰 Revenue", "🤖 AI Prediction", "🗄️ SQL Explorer", "📡 Power BI"
])

# ────────────────────────────────────────────────
#  TAB 1 — OVERVIEW
# ────────────────────────────────────────────────
with t1:
    sec("Key Metrics")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Orders",     f"{len(fdf):,}")
    c2.metric("Unique Customers", f"{fdf['customer_id'].nunique():,}")
    c3.metric("Delivered",        f"{(fdf['order_status']=='delivered').sum():,}")
    c4.metric("Avg Review",       f"{fdf['review_score'].mean():.2f} ⭐")
    c5.metric("Total Revenue",    f"R$ {fdf['payment_value'].sum():,.0f}")

    sec("Monthly Order Trend")
    mo = fdf.groupby("month").size().reset_index(name="n")
    fig,ax = plt.subplots(figsize=(11,3)); dark(fig,ax)
    ax.plot(mo["month"], mo["n"], color="#f97316", lw=2.5, marker="o", ms=5)
    ax.fill_between(mo["month"], mo["n"], alpha=.15, color="#f97316")
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], color="#64748b", fontsize=9)
    st.pyplot(fig, use_container_width=True); plt.close()

    col1,col2 = st.columns(2)
    with col1:
        sec("Order Status Split")
        sc = fdf["order_status"].value_counts()
        fig2,ax2 = plt.subplots(figsize=(5,4)); dark(fig2,ax2,ygrid=False)
        cols8 = ["#f97316","#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#ec4899","#06b6d4"]
        ax2.pie(sc.values, labels=sc.index, autopct="%1.1f%%", colors=cols8[:len(sc)],
                textprops={"color":"#e2e8f0","fontsize":9},
                wedgeprops={"linewidth":2,"edgecolor":"#1b1f2a"})
        st.pyplot(fig2, use_container_width=True); plt.close()

    with col2:
        sec("Orders by Weekday")
        wd = fdf["weekday"].value_counts().reindex(
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).fillna(0)
        fig3,ax3 = plt.subplots(figsize=(5,4)); dark(fig3,ax3)
        bc = ["#f97316" if i==wd.values.argmax() else "#3b82f6" for i in range(len(wd))]
        ax3.bar(wd.index, wd.values, color=bc, edgecolor="#1b1f2a", width=.6)
        plt.xticks(rotation=30, ha="right", color="#64748b", fontsize=8)
        st.pyplot(fig3, use_container_width=True); plt.close()

    sec("Hourly Order Pattern")
    hr = fdf.groupby("hour").size().reset_index(name="n")
    fig4,ax4 = plt.subplots(figsize=(11,2.5)); dark(fig4,ax4)
    ax4.fill_between(hr["hour"], hr["n"], color="#3b82f6", alpha=.4)
    ax4.plot(hr["hour"], hr["n"], color="#3b82f6", lw=2)
    ax4.set_xticks(range(0,24))
    st.pyplot(fig4, use_container_width=True); plt.close()

# ────────────────────────────────────────────────
#  TAB 2 — DELIVERY
# ────────────────────────────────────────────────
with t2:
    sec("Delivery Performance")
    dd = fdf.dropna(subset=["delivery_delay_days"])
    on  = (dd["delivery_delay_days"]<=0).sum()
    late= (dd["delivery_delay_days"]>0).sum()
    tot = on+late
    ad  = dd[dd["delivery_delay_days"]>0]["delivery_delay_days"].mean()
    ae  = dd[dd["delivery_delay_days"]<=0]["delivery_delay_days"].mean()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("On-Time",      f"{on:,}")
    c2.metric("Delayed",      f"{late:,}",
              delta=f"{late/tot*100:.1f}% rate" if tot else "N/A", delta_color="inverse")
    c3.metric("Avg Delay",    f"{ad:.1f} days" if not np.isnan(ad) else "N/A")
    c4.metric("Avg Early By", f"{abs(ae):.1f} days" if not np.isnan(ae) else "N/A")

    col1,col2 = st.columns(2)
    with col1:
        sec("Delay Distribution")
        fig5,ax5 = plt.subplots(figsize=(5,3.5)); dark(fig5,ax5)
        cl = dd["delivery_delay_days"].clip(-30,60)
        ax5.hist(cl[cl<=0], bins=20, color="#10b981", alpha=.85, label="Early/On-time")
        ax5.hist(cl[cl>0],  bins=20, color="#ef4444", alpha=.85, label="Delayed")
        ax5.legend(fontsize=8, labelcolor="#e2e8f0", facecolor="#1b1f2a")
        ax5.set_xlabel("Days (negative=early)", color="#64748b", fontsize=8)
        st.pyplot(fig5, use_container_width=True); plt.close()

    with col2:
        sec("Delay Rate by State (Top 10)")
        sd = (dd.groupby("customer_state")
              .apply(lambda x:(x["delivery_delay_days"]>0).mean()*100)
              .sort_values(ascending=False).head(10))
        fig6,ax6 = plt.subplots(figsize=(5,3.5)); dark(fig6,ax6,xgrid=True,ygrid=False)
        bc6 = ["#ef4444" if v>20 else "#f59e0b" if v>10 else "#10b981" for v in sd.values]
        ax6.barh(sd.index[::-1], sd.values[::-1], color=bc6[::-1], edgecolor="#1b1f2a")
        ax6.set_xlabel("Delay Rate %", color="#64748b", fontsize=8)
        st.pyplot(fig6, use_container_width=True); plt.close()

    sec("Review Score vs Avg Delay")
    rv2 = dd.groupby(dd["review_score"].round())["delivery_delay_days"].mean().reset_index()
    fig7,ax7 = plt.subplots(figsize=(11,3)); dark(fig7,ax7)
    ax7.bar(rv2["review_score"], rv2["delivery_delay_days"],
            color=["#10b981","#3b82f6","#f59e0b","#f97316","#ef4444"][:len(rv2)],
            edgecolor="#0d0f14", width=.5)
    ax7.axhline(0, color="#64748b", lw=1, linestyle="--")
    ax7.set_xlabel("Review Score", color="#64748b")
    ax7.set_ylabel("Avg Delay (days)", color="#64748b")
    st.pyplot(fig7, use_container_width=True); plt.close()

# ────────────────────────────────────────────────
#  TAB 3 — REVENUE
# ────────────────────────────────────────────────
with t3:
    sec("Revenue Overview")
    rdf = fdf.dropna(subset=["payment_value"])
    tot = rdf["payment_value"].sum()
    avg = rdf["payment_value"].mean()
    med = rdf["payment_value"].median()
    t10 = rdf.nlargest(max(1,int(len(rdf)*.1)),"payment_value")["payment_value"].sum()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Revenue",   f"R$ {tot:,.0f}")
    c2.metric("Avg Order Value", f"R$ {avg:.2f}")
    c3.metric("Median Order",    f"R$ {med:.2f}")
    c4.metric("Top 10% Rev",     f"R$ {t10:,.0f}")

    col1,col2 = st.columns(2)
    with col1:
        sec("Revenue by Month")
        mr = rdf.groupby("month")["payment_value"].sum()
        fig8,ax8 = plt.subplots(figsize=(5,3.5)); dark(fig8,ax8)
        ax8.bar(mr.index, mr.values, color="#f97316", edgecolor="#1b1f2a", width=.7)
        ax8.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"R${x/1000:.0f}k"))
        st.pyplot(fig8, use_container_width=True); plt.close()

    with col2:
        sec("Revenue by State (Top 10)")
        sr = rdf.groupby("customer_state")["payment_value"].sum().sort_values(ascending=False).head(10)
        fig9,ax9 = plt.subplots(figsize=(5,3.5)); dark(fig9,ax9,xgrid=True,ygrid=False)
        g = plt.cm.YlOrRd(np.linspace(.4,.9,len(sr)))
        ax9.barh(sr.index[::-1], sr.values[::-1], color=g[::-1], edgecolor="#1b1f2a")
        ax9.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"R${x/1000:.0f}k"))
        st.pyplot(fig9, use_container_width=True); plt.close()

    sec("Order Value Distribution")
    fig10,ax10 = plt.subplots(figsize=(11,2.5)); dark(fig10,ax10)
    ax10.hist(rdf["payment_value"].clip(0,500), bins=60, color="#3b82f6", edgecolor="#0d0f14", alpha=.85)
    ax10.axvline(avg, color="#f97316", lw=1.5, linestyle="--", label=f"Mean R${avg:.0f}")
    ax10.axvline(med, color="#10b981", lw=1.5, linestyle="--", label=f"Median R${med:.0f}")
    ax10.legend(fontsize=8, labelcolor="#e2e8f0", facecolor="#1b1f2a")
    ax10.set_xlabel("Payment Value R$ (capped 500)", color="#64748b", fontsize=8)
    st.pyplot(fig10, use_container_width=True); plt.close()

# ────────────────────────────────────────────────
#  TAB 4 — AI PREDICTION
# ────────────────────────────────────────────────
with t4:
    sec("Payment Value Predictor")

    if model is None:
        st.warning("""
**⚠️ Model not trained yet.**

Run this command in your terminal first:
```
python train_model.py
```
Wait for it to finish (2-4 min), then refresh this page.
        """)
    else:
        st.success(f"✅ Model ready — **ecommerce_model.pkl**  |  {len(feat)} features")
        st.caption("Enter order details below to predict the expected payment value.")
        st.markdown("---")

        col1,col2,col3 = st.columns(3)
        with col1:
            st.markdown("**📦 Product**")
            num_items     = st.number_input("Number of Items",       1,   50,    1)
            mean_price    = st.number_input("Avg Item Price (R$)",   0.0, 5000.0,100.0, step=10.0)
            total_freight = st.number_input("Freight Cost (R$)",     0.0, 500.0, 15.0,  step=1.0)
            mean_weight   = st.number_input("Product Weight (g)",    0.0, 30000.,500.0, step=50.0)

        with col2:
            st.markdown("**🗓️ Order Timing**")
            p_month   = st.selectbox("Purchase Month", range(1,13), index=5,
                            format_func=lambda x:["Jan","Feb","Mar","Apr","May","Jun",
                                                   "Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
            p_day     = st.number_input("Purchase Day", 1, 31, 15)
            p_hour    = st.slider("Purchase Hour", 0, 23, 14)
            p_weekday = st.selectbox("Weekday", range(0,7), index=1,
                            format_func=lambda x:["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])

        with col3:
            st.markdown("**💳 Payment**")
            installments = st.number_input("Installments",       1, 24, 1)
            n_pay_types  = st.number_input("Payment Types Used", 1,  4, 1)
            n_sellers    = st.number_input("Num Sellers",        1, 10, 1)
            review       = st.slider("Expected Review Score", 1.0, 5.0, 4.0, step=0.5)

        st.markdown("---")

        if st.button("🤖  Predict Payment Value"):
            raw = {
                "num_items":            num_items,
                "total_price":          mean_price * num_items,
                "mean_price":           mean_price,
                "total_freight":        total_freight,
                "num_unique_products":  min(num_items, 5),
                "num_unique_sellers":   n_sellers,
                "payment_installments": installments,
                "num_payment_types":    n_pay_types,
                "review_score":         review,
                "purchase_year":        2018,
                "purchase_month":       p_month,
                "purchase_day":         int(p_day),
                "purchase_hour":        p_hour,
                "purchase_weekday":     p_weekday,
                "mean_weight":          mean_weight,
                "mean_length":          20.0,
                "customer_state":       0,
            }
            inp = pd.DataFrame([raw])
            for c in feat:
                if c not in inp.columns: inp[c] = 0
            inp = inp[feat]

            pred      = float(model.predict(inp)[0])
            frt_pct   = total_freight / max(pred, 1) * 100
            net       = pred - total_freight

            st.markdown("### 📊 Result")
            r1,r2,r3 = st.columns(3)
            r1.markdown(f'<div class="pcard"><div class="lbl">Predicted Payment</div><div class="big" style="color:#f97316">R$ {pred:.2f}</div></div>', unsafe_allow_html=True)
            r2.markdown(f'<div class="pcard"><div class="lbl">Freight % of Order</div><div class="big" style="color:#3b82f6">{frt_pct:.1f}%</div></div>', unsafe_allow_html=True)
            r3.markdown(f'<div class="pcard"><div class="lbl">Est. Net Value</div><div class="big" style="color:#10b981">R$ {net:.2f}</div></div>', unsafe_allow_html=True)

            st.markdown("### 💡 Smart Insights")
            st.success(f"✅ Freight ratio: **{frt_pct:.1f}%**") if frt_pct<=30 else st.warning(f"⚠️ High freight ratio: **{frt_pct:.1f}%** — optimise logistics")
            if installments > 6: st.info(f"💳 {installments} installments — higher churn risk")
            if p_hour>=22 or p_hour<=5: st.warning("🌙 Night-time order — slight delay risk")
            if review < 3: st.error("⭐ Low review expected — check quality/logistics")
            elif review >= 4.5: st.success("⭐ High review expected — great customer experience")

# ────────────────────────────────────────────────
#  TAB 5 — SQL EXPLORER
# ────────────────────────────────────────────────
with t5:
    sec("Live SQL Query Engine")
    st.caption("Query the filtered **orders** table with any SQL statement.")

    conn = sqlite3.connect(":memory:")
    fdf.to_sql("orders", conn, index=False, if_exists="replace")

    PRESETS = {
        "-- pick a preset --": "",
        "Orders by Status":         "SELECT order_status, COUNT(*) AS total\nFROM orders\nGROUP BY order_status\nORDER BY total DESC",
        "Revenue by Month":         "SELECT month, ROUND(SUM(payment_value),2) AS revenue, COUNT(*) AS orders\nFROM orders\nGROUP BY month ORDER BY month",
        "Top 10 States by Revenue": "SELECT customer_state, ROUND(SUM(payment_value),2) AS revenue\nFROM orders\nGROUP BY customer_state ORDER BY revenue DESC LIMIT 10",
        "Avg Delay by State":       "SELECT customer_state, ROUND(AVG(delivery_delay_days),2) AS avg_delay\nFROM orders WHERE delivery_delay_days IS NOT NULL\nGROUP BY customer_state ORDER BY avg_delay DESC LIMIT 10",
        "Review Distribution":      "SELECT ROUND(review_score,0) AS score, COUNT(*) AS cnt\nFROM orders WHERE review_score IS NOT NULL\nGROUP BY score ORDER BY score",
        "Hourly Pattern":           "SELECT hour, COUNT(*) AS orders\nFROM orders GROUP BY hour ORDER BY hour",
        "High Value Orders":        "SELECT order_id, payment_value, customer_state, review_score\nFROM orders WHERE payment_value > 500\nORDER BY payment_value DESC LIMIT 20",
    }

    preset = st.selectbox("📋 Preset Queries", list(PRESETS.keys()))
    sql    = st.text_area("✏️ SQL Query", value=PRESETS[preset], height=130,
                          placeholder="SELECT * FROM orders LIMIT 10")

    if st.button("▶  Run Query"):
        if sql.strip():
            try:
                res = pd.read_sql(sql, conn)
                st.success(f"✅ {len(res):,} rows returned")
                st.dataframe(res, use_container_width=True)
                st.download_button("⬇️ Download CSV", res.to_csv(index=False),
                                   "result.csv", "text/csv")
            except Exception as e:
                st.error(f"❌ SQL Error: {e}")
        else:
            st.warning("Enter a query or pick a preset.")
    conn.close()

    st.markdown("**📌 Columns in `orders` table:**")
    st.code(", ".join(fdf.columns.tolist()), language="sql")

# ────────────────────────────────────────────────
#  TAB 6 — POWER BI
# ────────────────────────────────────────────────
with t6:
    sec("Power BI Integration")

    st.markdown("""
    <div style='background:#1b1f2a;border:1px solid #272c3a;border-radius:14px;padding:24px;margin-bottom:1.5rem;'>
      <div style='font-family:monospace;font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>How to embed your Power BI report</div>
      <div style='color:#e2e8f0;font-size:14px;line-height:1.8;'>
        1. Open your report in <b>Power BI Service</b> (app.powerbi.com)<br>
        2. Click <b>File → Embed report → Website or portal</b><br>
        3. Copy the <b>embed URL</b> (starts with https://app.powerbi.com/reportEmbed...)<br>
        4. Paste it in the box below and click <b>Load Report</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    pbi_url = st.text_input(
        "🔗 Paste your Power BI Embed URL here",
        placeholder="https://app.powerbi.com/reportEmbed?reportId=..."
    )

    if pbi_url and pbi_url.startswith("https://"):
        st.markdown(f"""
        <iframe
            src="{pbi_url}"
            width="100%"
            height="600"
            frameborder="0"
            allowfullscreen="true"
            style="border-radius:12px;border:1px solid #272c3a;">
        </iframe>
        """, unsafe_allow_html=True)
        st.success("✅ Power BI report embedded!")
    else:
        st.info("👆 Paste your Power BI embed URL above to display your report here.")

    st.markdown("---")
    sec("Power BI–Style KPI Summary")

    # KPI cards matching Power BI style
    rdf2 = fdf.dropna(subset=["payment_value"])
    dd2  = fdf.dropna(subset=["delivery_delay_days"])

    total_rev2   = rdf2["payment_value"].sum()
    avg_rev2     = rdf2["payment_value"].mean()
    total_orders = len(fdf)
    del_rate     = (fdf["order_status"] == "delivered").sum() / max(len(fdf), 1) * 100
    delay_rate2  = (dd2["delivery_delay_days"] > 0).sum() / max(len(dd2), 1) * 100
    avg_rev2_sc  = fdf["review_score"].mean()

    st.markdown(f"""
    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:2rem;'>
      <div style='background:#1b1f2a;border:1px solid #272c3a;border-left:4px solid #f97316;border-radius:10px;padding:20px;'>
        <div style='font-family:monospace;font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;'>Total Revenue</div>
        <div style='font-size:2rem;font-weight:800;color:#f97316;font-family:Syne,sans-serif;'>R$ {total_rev2:,.0f}</div>
        <div style='font-size:11px;color:#64748b;margin-top:4px;'>Across all filtered orders</div>
      </div>
      <div style='background:#1b1f2a;border:1px solid #272c3a;border-left:4px solid #3b82f6;border-radius:10px;padding:20px;'>
        <div style='font-family:monospace;font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;'>Total Orders</div>
        <div style='font-size:2rem;font-weight:800;color:#3b82f6;font-family:Syne,sans-serif;'>{total_orders:,}</div>
        <div style='font-size:11px;color:#64748b;margin-top:4px;'>Delivery rate: {del_rate:.1f}%</div>
      </div>
      <div style='background:#1b1f2a;border:1px solid #272c3a;border-left:4px solid #10b981;border-radius:10px;padding:20px;'>
        <div style='font-family:monospace;font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;'>Avg Order Value</div>
        <div style='font-size:2rem;font-weight:800;color:#10b981;font-family:Syne,sans-serif;'>R$ {avg_rev2:.2f}</div>
        <div style='font-size:11px;color:#64748b;margin-top:4px;'>Median customer spend</div>
      </div>
      <div style='background:#1b1f2a;border:1px solid #272c3a;border-left:4px solid #f59e0b;border-radius:10px;padding:20px;'>
        <div style='font-family:monospace;font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;'>Avg Review Score</div>
        <div style='font-size:2rem;font-weight:800;color:#f59e0b;font-family:Syne,sans-serif;'>{avg_rev2_sc:.2f} ⭐</div>
        <div style='font-size:11px;color:#64748b;margin-top:4px;'>Customer satisfaction index</div>
      </div>
      <div style='background:#1b1f2a;border:1px solid #272c3a;border-left:4px solid #ef4444;border-radius:10px;padding:20px;'>
        <div style='font-family:monospace;font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;'>Delay Rate</div>
        <div style='font-size:2rem;font-weight:800;color:#ef4444;font-family:Syne,sans-serif;'>{delay_rate2:.1f}%</div>
        <div style='font-size:11px;color:#64748b;margin-top:4px;'>Orders delivered late</div>
      </div>
      <div style='background:#1b1f2a;border:1px solid #272c3a;border-left:4px solid #8b5cf6;border-radius:10px;padding:20px;'>
        <div style='font-family:monospace;font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;'>On-Time Rate</div>
        <div style='font-size:2rem;font-weight:800;color:#8b5cf6;font-family:Syne,sans-serif;'>{100-delay_rate2:.1f}%</div>
        <div style='font-size:11px;color:#64748b;margin-top:4px;'>Orders delivered on time</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Power BI style bar chart — Revenue by State
    sec("Revenue by State — Power BI Style")
    sr2 = rdf2.groupby("customer_state")["payment_value"].sum().sort_values(ascending=True).tail(10)
    fig_pbi, ax_pbi = plt.subplots(figsize=(11, 4))
    dark(fig_pbi, ax_pbi, xgrid=True, ygrid=False)
    bars = ax_pbi.barh(sr2.index, sr2.values, color="#f97316", edgecolor="#0d0f14", height=0.6)
    # add value labels
    for bar, val in zip(bars, sr2.values):
        ax_pbi.text(val + total_rev2*0.001, bar.get_y() + bar.get_height()/2,
                    f"R${val/1000:.0f}k", va="center", color="#e2e8f0", fontsize=8,
                    fontfamily="monospace")
    ax_pbi.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x/1000:.0f}k"))
    ax_pbi.set_xlabel("Total Revenue", color="#64748b", fontsize=9)
    st.pyplot(fig_pbi, use_container_width=True); plt.close()

    # Download data for Power BI
    st.markdown("---")
    sec("Export Data for Power BI")
    st.markdown(
        "<span style='color:#64748b;font-family:monospace;font-size:12px'>"
        "Download the processed dataset to import directly into Power BI Desktop.</span>",
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns(3)

    orders_export = fdf[["order_id","order_status","month","year","hour","weekday",
                          "payment_value","customer_state","review_score",
                          "delivery_delay_days","num_items","total_freight","mean_price"]
                         ].dropna(subset=["payment_value"])

    col1.download_button(
        "⬇️ Download Orders CSV",
        data=orders_export.to_csv(index=False),
        file_name="olist_processed_orders.csv",
        mime="text/csv"
    )

    monthly_exp = fdf.groupby(["year","month"]).agg(
        orders=("order_id","count"),
        revenue=("payment_value","sum"),
        avg_review=("review_score","mean")
    ).reset_index()
    col2.download_button(
        "⬇️ Download Monthly Summary",
        data=monthly_exp.to_csv(index=False),
        file_name="olist_monthly_summary.csv",
        mime="text/csv"
    )

    state_exp = fdf.groupby("customer_state").agg(
        orders=("order_id","count"),
        revenue=("payment_value","sum"),
        delay_rate=("delivery_delay_days", lambda x: (x>0).mean()*100),
        avg_review=("review_score","mean")
    ).reset_index()
    col3.download_button(
        "⬇️ Download State Summary",
        data=state_exp.to_csv(index=False),
        file_name="olist_state_summary.csv",
        mime="text/csv"
    )