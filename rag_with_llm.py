"""
RAG Pipeline with LLM 
Adds: LLM-based answer generation with citations using Groq (free)
Flow: Question -> Retrieve chunks -> Send to LLM -> Cited answer
"""

import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# ---------- LOAD API KEY ----------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file. Please check your .env file.")


# ---------- STEP 1: Load All PDFs ----------
import glob
print("Step 1: Loading all PDFs...")
pdf_files = glob.glob("data/*.pdf")
documents = []
for pdf_path in pdf_files:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    documents.extend(docs)
    print(f"  Loaded: {pdf_path} ({len(docs)} pages)")
print(f"\nTotal pages loaded: {len(documents)}\n")

# ---------- STEP 2: Chunk ----------
print("Step 2: Chunking...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.\n")

# ---------- STEP 3: Embed + Store ----------
print("Step 3: Loading vector store...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)
print("Vector store ready.\n")

# ---------- STEP 4: Set up LLM ----------
print("Step 4: Connecting to Groq LLM...")
llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)
print("LLM ready.\n")

# ---------- STEP 5: RAG Function ----------
def ask(question):
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)

    # Retrieve top 3 relevant chunks
    results = vectorstore.similarity_search(question, k=3)

    # Build context string with page citations
    context = ""
    sources = []
    for i, doc in enumerate(results, 1):
        page = doc.metadata.get('page', 'N/A')
        context += f"\n[Source {i} - Page {page}]:\n{doc.page_content}\n"
        sources.append(f"Page {page}")

    # Send to LLM
    system_prompt = """You are a research assistant helping answer questions about academic papers.
Answer the question using ONLY the provided context.
Always cite which source (Page number) supports each part of your answer.
If the context does not contain enough information, say: 
'The available documents do not contain sufficient information to answer this question.'
Never make up information."""

    user_prompt = f"""Context from the paper:
{context}

Question: {question}

Provide a clear, cited answer based only on the context above."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)

    print(f"\nAnswer:\n{response.content}")
    print(f"\nSources used: {', '.join(sources)}")
    return response.content

# ---------- STEP 6: Test with sample questions ----------
ask("What are reflection tokens in Self-RAG?")
ask("How does Self-RAG decide when to retrieve documents?")
ask("What datasets were used to evaluate Self-RAG?")