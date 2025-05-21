import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

df = pd.read_csv("data/enriched_vendor_summaries.csv")
summaries = df["Summary"].dropna().tolist()

splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=20)
documents = splitter.create_documents(summaries)

embedding = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embedding)
db.save_local("vectorstore")

print("✅ Vectorstore saved.")
