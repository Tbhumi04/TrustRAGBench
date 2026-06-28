"""
RAG Pipeline - Week 2 Baseline
Steps: Load PDF -> Chunk text -> Generate embeddings -> Store in vector DB -> Retrieve relevant chunks
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---------- STEP 1: Load the PDF ----------
print("Step 1: Loading PDF...")
loader = PyPDFLoader("data/self_rag_paper.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages from the PDF.\n")

# ---------- STEP 2: Split into chunks ----------
print("Step 2: Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.\n")

# ---------- STEP 3: Generate embeddings + store in vector DB ----------
print("Step 3: Generating embeddings and storing in ChromaDB...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)
print("Vector database created and saved locally.\n")

# ---------- STEP 4: Test retrieval ----------
print("Step 4: Testing retrieval...")
query = "What are reflection tokens in Self-RAG?"
results = vectorstore.similarity_search(query, k=3)

print(f"\nQuery: {query}")
print(f"\nTop {len(results)} retrieved chunks:\n")
for i, doc in enumerate(results, 1):
    print(f"--- Result {i} (Page {doc.metadata.get('page', 'N/A')}) ---")
    print(doc.page_content[:300])
    print()
    