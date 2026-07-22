"""
config_comparison.py
Compares RAG performance across different chunking configurations.
Tests chunk sizes: 250, 500, 1000 characters.
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
import shutil

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ---------- SETUP ----------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.0
)

# ---------- TEST QUESTIONS ----------
test_questions = [
    "What are reflection tokens in Self-RAG?",
    "What is HyDE and how does it improve retrieval?",
    "What is chain of thought prompting?",
    "What architecture does Mistral 7B use?",
    "How does ReAct combine reasoning and acting?",
    "How does Sentence-BERT generate sentence embeddings?",
    "What is dense passage retrieval in RAG?",
]

# Retrieval keywords for Precision@k
test_set_keywords = [
    ["reflection token", "critique", "retrieval token"],
    ["hypothetical", "hyde", "document embeddings"],
    ["chain of thought", "reasoning", "step by step"],
    ["sliding window", "grouped query", "mistral"],
    ["react", "reasoning", "acting", "action"],
    ["siamese", "sentence embedding", "bert"],
    ["dense", "passage retrieval", "dpr"],
]

# ---------- FUNCTIONS ----------
def load_all_pdfs():
    pdf_files = glob.glob("data/*.pdf")
    documents = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    return documents

def build_vectorstore(documents, chunk_size, chunk_overlap, db_path):
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_path
    )
    return vectorstore, len(chunks)

def get_answer(question, vectorstore, k=6):
    results = vectorstore.similarity_search(question, k=k)
    context = ""
    contexts = []
    for i, doc in enumerate(results, 1):
        page = doc.metadata.get('page', 'N/A')
        context += f"\n[Source {i} - Page {page}]:\n{doc.page_content}\n"
        contexts.append(doc.page_content)
    system_prompt = """You are a research assistant. Answer using ONLY the provided context.
If context is insufficient, say: 'The available documents do not contain sufficient information.'
Never fabricate information."""
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a cited answer."
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    return response.content, context

def score_faithfulness(answer, context):
    prompt = f"""Context: {context[:2000]}
Answer: {answer}
Is every claim in the answer supported by the context?
Reply ONLY with a number 0.0 to 1.0. Nothing else."""
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        return min(max(float(response.content.strip()), 0.0), 1.0)
    except:
        return 0.5

def score_relevancy(question, answer):
    prompt = f"""Question: {question}
Answer: {answer}
How well does this answer address the question?
Reply ONLY with a number 0.0 to 1.0. Nothing else."""
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        return min(max(float(response.content.strip()), 0.0), 1.0)
    except:
        return 0.5

def calc_precision(results, keywords, k):
    flags = [any(kw.lower() in doc.page_content.lower() for kw in keywords)
             for doc in results]
    return sum(flags) / k

def evaluate_config(vectorstore, k=6):
    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []

    for i, q in enumerate(test_questions):
        answer, context = get_answer(q, vectorstore, k)
        f_score = score_faithfulness(answer, context)
        r_score = score_relevancy(q, answer)
        results = vectorstore.similarity_search(q, k=k)
        p_score = calc_precision(results, test_set_keywords[i], k)
        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)
        precision_scores.append(p_score)
        print(f"  ✓ Q{i+1}: Faith={f_score:.2f} | Rel={r_score:.2f} | Prec={p_score:.2f}")

    return {
        "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
        "relevancy": sum(relevancy_scores) / len(relevancy_scores),
        "precision": sum(precision_scores) / len(precision_scores),
    }

# ---------- RUN CONFIGS ----------
print("Loading all PDFs...")
documents = load_all_pdfs()
print(f"Loaded {len(documents)} pages total.\n")

configs = [
    {"chunk_size": 250,  "chunk_overlap": 25,  "k": 6},
    {"chunk_size": 500,  "chunk_overlap": 50,  "k": 6},
    {"chunk_size": 1000, "chunk_overlap": 100, "k": 6},
]

all_results = []

for cfg in configs:
    cs = cfg["chunk_size"]
    co = cfg["chunk_overlap"]
    k  = cfg["k"]
    db_path = f"chroma_db_chunk{cs}"

    print(f"\n{'='*60}")
    print(f"CONFIG: chunk_size={cs}, overlap={co}, k={k}")
    print(f"{'='*60}")

    vectorstore, num_chunks = build_vectorstore(documents, cs, co, db_path)
    print(f"Built vector store: {num_chunks} chunks")
    print("Evaluating...")

    scores = evaluate_config(vectorstore, k)
    scores["chunk_size"] = cs
    scores["num_chunks"] = num_chunks
    all_results.append(scores)

# ---------- RESULTS TABLE ----------
print(f"\n\n{'='*70}")
print("CONFIGURATION COMPARISON RESULTS")
print(f"{'='*70}")
print(f"{'Chunk Size':<12} {'# Chunks':<12} {'Faithfulness':<15} {'Relevancy':<12} {'Precision@6'}")
print("-"*70)
for r in all_results:
    print(f"{r['chunk_size']:<12} {r['num_chunks']:<12} "
          f"{r['faithfulness']:.4f} ({r['faithfulness']*100:.1f}%)   "
          f"{r['relevancy']:.4f} ({r['relevancy']*100:.1f}%)   "
          f"{r['precision']:.4f} ({r['precision']*100:.1f}%)")
print("="*70)

best = max(all_results, key=lambda x: x['faithfulness'])
print(f"\nBest config: chunk_size={best['chunk_size']} "
      f"with faithfulness={best['faithfulness']*100:.1f}%")