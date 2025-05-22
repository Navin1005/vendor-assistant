import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

forecast_df = pd.read_csv("data/simulated_forecast.csv")
forecast_df["Item"] = forecast_df["Item"].str.lower()

embedding = OpenAIEmbeddings()
db = FAISS.load_local("vectorstore", embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

def rag_query(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["how much", "quantity", "do we need", "should i order", "forecast", "demand"]):
        for item in forecast_df["Item"]:
            if item in q:
                qty = forecast_df.loc[forecast_df["Item"] == item, "Quantity"].values[0]
                qty_lbs = round(qty * 2.20462, 2)
                return (
                    f"📦 **AI Forecast Insight**\n\n"
                    f"🔹 Projected demand for **{item.title()}** next week: **{qty_lbs} lbs**.\n"
                    f"💼 Helps optimize bulk ordering and reduce overstock waste."
                )
        return "⚠️ No forecast data found for that item."
    return qa_chain.run(query)
