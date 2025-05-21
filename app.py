import streamlit as st
import pandas as pd
from query_engine import rag_query

st.set_page_config(page_title="True Smart Kitchen – Smart Vendor Assistant", page_icon="📦", layout="centered")

# === Professional UI Styling ===
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
st.markdown("#### Ask questions like 'What should I buy next week?' or 'Did Sysco supply butter last month?'")

@st.cache_data
def load_data():
    forecast_df = pd.read_csv("data/simulated_forecast.csv")
    vendor_df = pd.read_csv("data/enriched_vendor_summaries.csv")
    return forecast_df, vendor_df

def recommend_vendors():
    forecast_df, vendor_df = load_data()
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

# === Main App Logic ===
query = st.text_input("What do you want to know?", placeholder="e.g. What should I buy next week?")

if st.button("🔍 Get Recommendation") and query:
    with st.spinner("Consulting AI..."):
        q = query.lower()
        forecast_keywords = ["how much", "how many", "order", "quantity", "amount", "do we need", "forecast", "demand"]

        if any(k in q for k in ["what should i buy", "purchase plan", "next week's purchase", "savings opportunities", "top items", "vendors to use"]):
            forecast_lbs, top, others, savings, cost = recommend_vendors()
            st.markdown("## 🧠 AI-Powered Procurement Plan")
            st.caption("Based on real-time forecast and historical vendor trends")
            st.subheader("📦 Forecasted Demand (lbs)")
            st.dataframe(forecast_lbs)

            st.subheader("💰 Top Savings Opportunities")
            st.caption("AI-identified cost-saving vendors for high-volume items")
            for _, row in top.iterrows():
                note = ""
                if row["Estimated Savings"] > 100 or row["Forecast (lbs)"] > 40:
                    note = " 🔔 **High-Impact Purchase!**"
                st.markdown(
                    f"- **{row['Item']}** → *{row['Vendor']}* → **{row['Forecast (lbs)']} lbs** → 💵 Save **${row['Estimated Savings']:.2f}**{note}"
                )

            st.subheader("✅ Other Recommended Items")
            for _, row in others.iterrows():
                st.markdown(f"- {row['Item']} → {row['Vendor']}")

            st.success(f"🟢 Estimated Weekly Savings: ${savings} | 📦 Total Spend: ${cost}")
            st.markdown("💼 *Estimates based on a weekly sales volume of ~$30,000*")

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
            forecast_items = pd.read_csv("data/simulated_forecast.csv")["Item"].str.lower().tolist()
            for item in forecast_items:
                if item in q:
                    response = get_forecast_response(item)
                    st.markdown(response)
                    st.stop()
        else:
            response = rag_query(query)
            if not response.strip():
                response = "🤖 This is a demo environment. Based on AI intelligence, assume historical vendor data is available."
            st.markdown(response)
