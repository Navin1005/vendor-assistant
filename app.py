import streamlit as st
import pandas as pd
import os
import random
from datetime import datetime, timedelta
from query_engine import rag_query

st.set_page_config(page_title="True Smart Kitchen – Smart Vendor Assistant", page_icon="📦", layout="wide")

# === CSS Styling ===
st.markdown("""
    <style>
        html, body {
            background-color: #F5F8FA;
            font-family: "Segoe UI", sans-serif;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1000px;
            margin: auto;
        }
        h1, h2, h3 {
            color: #102A43;
        }
        .center {
            text-align: center;
        }
        .stButton>button {
            background-color: #0061A8;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
        }
        .stButton>button:hover {
            background-color: #004e86;
        }
    </style>
""", unsafe_allow_html=True)

# === Load Data ===
@st.cache_data
def load_data():
    forecast_df = pd.read_csv("data/simulated_forecast.csv")
    vendor_df = pd.read_csv("data/enriched_vendor_summaries.csv")
    if "monthly_sales.csv" in os.listdir("data"):
        sales_df = pd.read_csv("data/monthly_sales.csv", encoding="utf-8-sig")
        sales_df.columns = sales_df.columns.str.strip().str.replace('\ufeff', '')
    else:
        sales_df = None
    return forecast_df, vendor_df, sales_df

forecast_df, vendor_df, sales_df = load_data()

# === Navigation State ===
if "page" not in st.session_state:
    st.session_state.page = "Welcome"

# === Welcome Page ===
if st.session_state.page == "Welcome":
    st.markdown("<h1 class='center'>📦 True Smart Kitchen – Smart Vendor Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='center'>Your AI-powered assistant for smarter inventory forecasting and vendor selection</h3>", unsafe_allow_html=True)
    st.image("data/ChatGPT Image May 19, 2025, 08_34_31 PM.png", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("➡️ Go to Dashboard"):
        st.session_state.page = "Dashboard"
        st.rerun()

# === Dashboard Page ===
elif st.session_state.page == "Dashboard":
    st.title("📊 AI-Powered Kitchen Analytics Dashboard")
    st.subheader("📈 Last 6 Months Sales Overview")
    if sales_df is not None:
        monthly_sales = sales_df.groupby("Month")["Sales"].sum().reset_index()
        st.bar_chart(monthly_sales.set_index("Month"))

        recent_sales = monthly_sales["Sales"].tail(3).values
        if len(recent_sales) >= 2:
            growth = (recent_sales[-1] - recent_sales[-2])
            forecast_next = int(recent_sales[-1] + growth)
            next_month = pd.to_datetime(monthly_sales["Month"].iloc[-1]) + pd.DateOffset(months=1)
            st.markdown(f"### 🔮 Forecast for Next Month ({next_month.strftime('%Y-%m')})")
            st.success(f"Projected Sales: **${forecast_next:,}** based on last trend 📈")

        st.subheader("🏆 Top 5 Selling Items")
        top_items = sales_df.groupby("Item")["Sales"].sum().sort_values(ascending=False).head(5)
        st.table(top_items.reset_index().rename(columns={"Sales": "Total Sales"}))

    st.markdown("### 📌 Technical Highlights")
    st.markdown("""
    - 🔍 **Data Cleaning** with `pandas` to clean vendor invoices, normalize item names, unify units, and handle missing values and webscraping  
    - 🧠 **Forecast Algorithms**: ARIMA for univariate time series prediction on items with consistent historical patterns  
    - 📊 **Vendor Matching** Selected lowest average price per item using historical vendor performance  
    - 🤖 **AI Q&A** Integrated LangChain + OpenAI GPT to answer natural language questions using contextual vendor summaries  
    - 💼 **Interactive UI** built with Streamlit
    """)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("➡️ Go to Forecast & Vendor Plan"):
        st.session_state.page = "Forecast"
        st.rerun()

# === Forecast Page ===
elif st.session_state.page == "Forecast":
    st.title("🔮 Forecasting & Vendor Plan Assistant")

    def recommend_vendors():
        forecast_df["Quantity"] = pd.to_numeric(forecast_df["Quantity"], errors="coerce") * 3
        forecast_df["Forecast (lbs)"] = (forecast_df["Quantity"] * 2.20462).round(2)
        parsed = vendor_df["Summary"].str.extract(
            r"(?P<Date>.+), (?P<Vendor>.+) sold (?P<Quantity>\d+) unit\(s\) of (?P<Item>.+) at \$(?P<Price>[\d\.]+)"
        )
        parsed["Price"] = parsed["Price"].astype(float)

        item_records = {}
        total_cost, total_savings = 0.0, 0.0
        for _, row in forecast_df.iterrows():
            item, qty = row["Item"], float(row["Quantity"])
            item_data = parsed[parsed["Item"].str.contains(item, case=False, na=False)]
            if item_data.empty or qty == 0:
                continue
            avg_price = item_data.groupby("Vendor")["Price"].mean().reset_index()
            if len(avg_price) <= 1:
                continue
            best = avg_price.sort_values("Price").iloc[0]
            vendor = best["Vendor"]
            price = best["Price"]
            avg_all = item_data["Price"].mean()
            savings = max((avg_all - price) * qty, 0)
            cost = qty * price
            total_cost += cost
            total_savings += savings
            item_records[item] = {
                "Item": item,
                "Quantity": qty,
                "Forecast (lbs)": round(qty * 2.20462, 2),
                "Vendor": vendor,
                "Unit Price": price,
                "Estimated Savings": round(savings, 2),
                "Total Cost": round(cost, 2)
            }

        df = pd.DataFrame(item_records.values())
        top = df[df["Estimated Savings"] > 5].sort_values("Estimated Savings", ascending=False).head(10)
        others = df[df["Estimated Savings"] <= 5]
        return df[["Item", "Forecast (lbs)"]], top, others, round(total_savings, 2), round(total_cost, 2)

    def get_forecast_response(item_name):
        forecast_df["Item"] = forecast_df["Item"].str.lower()
        forecast_df["Quantity"] = pd.to_numeric(forecast_df["Quantity"], errors="coerce") * 3
        forecast_df["Forecast (lbs)"] = (forecast_df["Quantity"] * 2.20462).round(2)
        match = forecast_df[forecast_df["Item"] == item_name.lower()]
        if not match.empty:
            qty_lbs = match.iloc[0]["Forecast (lbs)"]
            return f"📦 Forecasted demand for **{item_name.title()}** next week: **{qty_lbs} lbs**"
        else:
            return f"⚠️ Forecast not found for **{item_name.title()}**"

    query = st.text_input("What do you want to know?", placeholder="e.g. What should I buy next week?")
    if st.button("🔍 Get Recommendation") and query:
        with st.spinner("Consulting AI..."):
            q = query.lower()
            if any(k in q for k in ["what should i buy", "purchase plan", "savings opportunities"]):
                forecast_lbs, top, others, savings, cost = recommend_vendors()
                st.subheader("📦 Forecasted Demand (lbs)")
                st.dataframe(forecast_lbs)

                st.subheader("💰 Top Savings")
                for _, row in top.iterrows():
                    st.markdown(f"- **{row['Item']}** → *{row['Vendor']}* → {row['Forecast (lbs)']} lbs → 💵 Save ${row['Estimated Savings']}")

                st.subheader("✅ Other Recommended Items")
                for _, row in others.iterrows():
                    st.markdown(f"- {row['Item']} → {row['Vendor']}")

                st.success(f"Estimated Savings: ${savings} | Total Spend: ${cost}")
            else:
                response = rag_query(query)
                st.markdown(response)

    # === Place Order Simulation ===
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("🛒 Finalize Your Purchase")
    if st.button("✅ Place Order Now"):
        order_id = f"ORD{random.randint(10000, 99999)}"
        delivery_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        total_amount = round(random.uniform(2000, 2500), 2)

        st.success("Order Placed Successfully!")
        st.markdown(f"""
        - 🧾 **Order ID**: `{order_id}`  
        - 🚚 **Expected Delivery Date**: `{delivery_date}`  
        - 💵 **Total Amount**: `${total_amount}`
        """)

    if st.button("⬅️ Back to Dashboard"):
        st.session_state.page = "Dashboard"
        st.rerun()
