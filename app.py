import streamlit as st
import pandas as pd
import os
from query_engine import rag_query

st.set_page_config(page_title="True Smart Kitchen – Smart Vendor Assistant", page_icon="📦", layout="wide")

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
        .stButton > button {
            background-color: #0061A8;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            border: none;
            transition: background-color 0.2s ease;
        }
        .stButton > button:hover {
            background-color: #004e86;
        }
        .stTextInput > div > div > input {
            font-size: 1.05rem;
            padding: 0.5rem;
            border-radius: 6px;
        }
        .stMarkdown p {
            font-size: 1.05rem;
            color: #334E68;
        }
        .stDataFrame {
            background-color: white;
            border: 1px solid #d9d9d9;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        }
    </style>
""", unsafe_allow_html=True)

st.title("📦 True Smart Kitchen – Smart Vendor Assistant")

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

def recommend_vendors():
    forecast_df, vendor_df, _ = load_data()
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
    forecast_df = pd.read_csv("data/simulated_forecast.csv")
    forecast_df["Item"] = forecast_df["Item"].str.lower()
    forecast_df["Quantity"] = pd.to_numeric(forecast_df["Quantity"], errors="coerce") * 3
    forecast_df["Forecast (lbs)"] = (forecast_df["Quantity"] * 2.20462).round(2)

    match = forecast_df[forecast_df["Item"] == item_name.lower()]
    if not match.empty:
        qty_lbs = match.iloc[0]["Forecast (lbs)"]
        return (
            f"📦 **AI Forecast Insight**\n\n"
            f"🔹 Forecasted demand for **{item_name.title()}**: **{qty_lbs} lbs** next week.\n\n"
            f"💼 This projection helps optimize ordering and reduce inventory waste based on demand."
        )
    else:
        return f"⚠️ Forecast not found for **{item_name.title()}**. Try asking about another item."

# === Tabs ===
tabs = st.tabs(["📊 Dashboard", "📦 Inventory Forecast & Vendor Plan"])

# === Tab 1: Dashboard ===
forecast_df, vendor_df, sales_df = load_data()
with tabs[0]:
    st.subheader("📈 Last 6 Months Sales Overview")
    if sales_df is not None:
        monthly_sales = sales_df.groupby("Month")["Sales"].sum().reset_index()
        st.bar_chart(monthly_sales.set_index("Month"))

        recent_sales = monthly_sales["Sales"].tail(3).values
        if len(recent_sales) >= 2:
            growth = (recent_sales[-1] - recent_sales[-2])
            forecast_next = int(recent_sales[-1] + growth)
            next_month = pd.to_datetime(monthly_sales["Month"].iloc[-1]) + pd.DateOffset(months=1)
            next_month_label = next_month.strftime("%Y-%m")
            st.markdown(f"### 🔮 Forecast for Next Month ({next_month_label})")
            st.success(f"Projected Sales: **${forecast_next:,}** based on last trend 📈")
    else:
        st.warning("No sales data available.")

    st.subheader("🏆 Top 5 Selling Items")
    if sales_df is not None:
        top_items = sales_df.groupby("Item")["Sales"].sum().sort_values(ascending=False).head(5)
        st.table(top_items.reset_index().rename(columns={"Sales": "Total Sales"}))
    else:
        st.warning("No item sales data available.")

    st.markdown("---")
    st.markdown("### 📌 Technical Highlights")
    st.markdown("""
    - 🔍 **Data Cleaning**: Used `pandas` to clean and structure historical vendor and forecast datasets.
    - 📦 **Forecast Modeling**: Applied scaled historical demand and converted units to pounds (lbs) for easy ordering.
    - 🧠 **LLM Integration**: Used OpenAI + LangChain RAG to handle vendor-related Q&A.
    - 💰 **Optimization Engine**: Matched forecast demand to cheapest suppliers using price history.
    - 📊 **Professional UI**: Streamlit dashboard with forecasting, cost savings, and business reporting.
    """)

# === Tab 2: Forecast & Vendor Plan ===
with tabs[1]:
    query = st.text_input("What do you want to know?", placeholder="e.g. What should I buy next week?")

    if st.button("🔍 Get Recommendation") and query:
        with st.spinner("Consulting AI..."):
            q = query.lower()
            forecast_keywords = ["how much", "how many", "order", "quantity", "amount", "do we need", "forecast", "demand"]

            if any(k in q for k in ["what should i buy", "purchase plan", "next week's purchase", "savings opportunities", "top items", "vendors to use"]):
                forecast_lbs, top, others, savings, cost = recommend_vendors()
                st.markdown("## 🧠 AI-Powered Procurement Plan")
                st.subheader("📦 Forecasted Demand (lbs)")
                st.dataframe(forecast_lbs)

                st.subheader("💰 Top Savings Opportunities")
                for _, row in top.iterrows():
                    note = " 🔔 **High-Impact Purchase!**" if row["Estimated Savings"] > 100 or row["Forecast (lbs)"] > 40 else ""
                    st.markdown(f"- **{row['Item']}** → *{row['Vendor']}* → **{row['Forecast (lbs)']} lbs** → 💵 Save **${row['Estimated Savings']:.2f}**{note}")

                st.subheader("✅ Other Recommended Items")
                for _, row in others.iterrows():
                    st.markdown(f"- {row['Item']} → {row['Vendor']}")

                st.success(f"🟢 Estimated Weekly Savings: ${savings} | 📦 Total Spend: ${cost}")

                if savings >= 1000:
                    st.markdown(f"""
                    ### 📈 Profitability Forecast
                    Your top 10 items this week will save you **${savings}** if you follow this plan.

                    ✅ Forecasted demand matched with cheapest suppliers  
                    ✅ Real-time quantity planning in **lbs**  
                    ✅ Profit margin protected before the week even starts

                    💡 *This makes your kitchen not just smarter — but more profitable every week.*
                    """)
            elif any(k in q for k in forecast_keywords):
                for item in forecast_df["Item"].str.lower():
                    if item in q:
                        response = get_forecast_response(item)
                        st.markdown(response)
                        st.stop()
            else:
                response = rag_query(query)
                if not response.strip():
                    response = "🤖 This is a demo environment. Based on AI intelligence, assume historical vendor data is available."
                st.markdown(response)
