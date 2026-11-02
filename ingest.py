"""Run once after putting your paper/report (.pdf/.txt/.md) in docs/:  python ingest.py"""
from app.rag import RagEngine

if __name__ == "__main__":
    n = RagEngine().ingest()
    print(f"Ingested {n} chunks into ChromaDB.")
