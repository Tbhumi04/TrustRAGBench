from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

queries = [
    "What is dense passage retrieval in RAG?",
    "How does CRAG correct poor retrievals?",
    "What are the key contributions of LLaMA 2?",
]

for query in queries:
    print(f"\nQuery: {query}")
    print("="*60)
    results = vectorstore.similarity_search(query, k=6)
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "N/A")
        page = doc.metadata.get("page", "N/A")
        print(f"Result {i} - Page {page} - File: {source}")
        print(doc.page_content[:200])
        print()