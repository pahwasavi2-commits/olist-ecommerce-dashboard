# 🛒 Olist E-Commerce Intelligence Dashboard

> A full-stack data science project built on the Brazilian Olist E-Commerce public dataset.  
> Features ML prediction, interactive analytics, SQL explorer, and Power BI integration.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow?style=flat-square&logo=powerbi)

---

## 📸 Features

| Feature | Description |
|---|---|
| 📊 Overview | KPIs, monthly trends, weekday & hourly patterns |
| 🚚 Delivery Analysis | On-time vs delayed, state-wise delay rates, review correlation |
| 💰 Revenue Analytics | Total/avg/median revenue, state & month breakdown |
| 🤖 AI Prediction | Random Forest model predicts payment value |
| 🗄️ SQL Explorer | Live SQL queries with 7 presets + CSV export |
| 📡 Power BI | Embed Power BI reports + export cleaned CSVs |

---

## 📁 Project Structure

```
archive (1)/
│
├── app.py                          ← Streamlit dashboard
├── train_model.py                  ← ML model training script
├── ecommerce_model.pkl             ← Trained Random Forest model
├── model_features.pkl              ← Feature list for prediction
├── requirements.txt                ← Python dependencies
│
├── olist_orders_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_customers_dataset.csv
├── olist_products_dataset.csv
├── olist_sellers_dataset.csv
└── product_category_name_translation.csv
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/olist-ecommerce-dashboard.git
cd olist-ecommerce-dashboard
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Get the Olist dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place all CSV files in the project folder.

### 5. Train the model
```bash
python train_model.py
```

### 6. Run the dashboard
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🤖 ML Model Details

- **Algorithm:** Random Forest Regressor
- **Target:** `payment_value` (order payment amount in BRL)
- **Features:** 17 engineered features from 6 CSV files
- **Anti-leakage:** Delivery timestamps removed from training
- **Performance:** R² ≈ 0.85–0.92 on test set

### Features Used
| Feature | Source |
|---|---|
| num_items, total_price, mean_price | order_items |
| total_freight, num_unique_products | order_items |
| payment_installments, num_payment_types | payments |
| review_score | reviews |
| purchase_month, hour, weekday | orders |
| mean_weight, mean_length | products |
| customer_state | customers |

---

## 📊 Dataset

- **Source:** [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- **Size:** ~100k orders, 8 CSV files
- **Period:** 2016–2018

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — Interactive web dashboard
- **scikit-learn** — Random Forest ML model
- **pandas / numpy** — Data processing
- **matplotlib** — Custom dark-theme charts
- **SQLite** — In-memory SQL queries
- **joblib** — Model serialization
- **Power BI** — Business intelligence integration

---

## 👤 Author

**Savi Pahwa**  
📧 pahwasavi2@gmail.com 
🔗 [LinkedIn](https://linkedin.com/in/Savi Pahwa)  
🐙 [GitHub](https://github.com/SAVI_PAHWA)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
