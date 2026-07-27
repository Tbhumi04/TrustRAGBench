"""
app.py
TrustRAGBench — Interactive RAG Dashboard
Streamlit web interface for research paper Q&A with evaluation scores.
"""

import os
import glob
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="TrustRAGBench",
    page_icon="📚",
    layout="wide"
)

# ---------- HEADER ----------
st.title("📚 TrustRAGBench")
st.caption("AI Research Paper Q&A · Retrieval-Augmented Generation with Evaluation")
st.divider()

# ---------- SIDEBAR CONFIG ----------
st.sidebar.header("⚙️ Configuration")
chunk_size = st.sidebar.selectbox(
    "Chunk size",
    options=[250, 500, 1000],
    index=1,
    help="Size of text chunks. Larger = better context, smaller = better precision."
)
top_k = st.sidebar.selectbox(
    "Top-k retrieval",
    options=[3, 6, 10],
    index=1,
    help="Number of chunks to retrieve per query."
)
st.sidebar.divider()

# ---------- SIDEBAR METRICS ----------
st.sidebar.subheader("📊 Evaluation Results")
st.sidebar.metric("Precision@3", "90.0%", "+20% after query refinement")
st.sidebar.metric("Faithfulness (chunk=1000)", "68.6%", "+12.9% vs baseline")
st.sidebar.metric("Safety refusal rate", "100%", "7/7 out-of-scope refused")
st.sidebar.divider()

# ---------- SIDEBAR PAPERS ----------
st.sidebar.subheader("📄 Indexed Papers")
papers = [
    "Self-RAG (Asai et al., 2023)",
    "RAG Original (Lewis et al., 2020)",
    "CRAG (Yan et al., 2024)",
    "HyDE (Gao et al., 2022)",
    "LLaMA 2 (Touvron et al., 2023)",
    "Mistral 7B (Jiang et al., 2023)",
    "Chain-of-Thought (Wei et al., 2022)",
    "RAGAS (Es et al., 2023)",
    "Sentence-BERT (Reimers et al., 2019)",
    "ReAct (Yao et al., 2022)",
]
for p in papers:
    st.sidebar.caption(f"• {p}")

# ---------- LOAD MODELS (cached) ----------
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

@st.cache_resource
def load_documents():
    pdf_files = glob.glob("data/*.pdf")
    documents = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    return documents

# ---------- BUILD VECTOR STORE ----------
@st.cache_resource
def build_vectorstore(_docs, chunk_size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = splitter.split_documents(_docs)
    embedding_model = load_embedding_model()
    db_path = f"chroma_db_chunk{chunk_size}"
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model
    )
    return vectorstore, len(chunks)

# ---------- SCORE FUNCTIONS ----------
def score_faithfulness(answer, context, llm):
    prompt = f"""Context: {context[:2000]}
Answer: {answer}
Is every claim in the answer supported by the context?
Reply ONLY with a number 0.0 to 1.0. Nothing else."""
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        return min(max(float(response.content.strip()), 0.0), 1.0)
    except:
        return 0.5

def score_relevancy(question, answer, llm):
    prompt = f"""Question: {question}
Answer: {answer}
How well does this answer address the question?
Reply ONLY with a number 0.0 to 1.0. Nothing else."""
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        return min(max(float(response.content.strip()), 0.0), 1.0)
    except:
        return 0.5

# ---------- LOAD EVERYTHING ----------
with st.spinner("Loading models and documents..."):
    embedding_model = load_embedding_model()
    llm = load_llm()
    documents = load_documents()
    overlap = chunk_size // 10
    vectorstore, num_chunks = build_vectorstore(documents, chunk_size, overlap)

# ---------- STATS ROW ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Papers indexed", "10", "257 pages")
col2.metric("Chunks created", f"{num_chunks:,}", f"chunk size = {chunk_size}")
col3.metric("Embedding model", "MiniLM-L6-v2", "384 dimensions")
col4.metric("LLM", "LLaMA 3.1 8B", "via Groq API")

st.divider()

# ---------- MAIN TABS ----------
tab1, tab2, tab3 = st.tabs(["💬 Ask a Question", "📊 Config Comparison", "🛡️ Safety Evaluation"])

# ========== TAB 1: Q&A ==========
with tab1:
    question = st.text_input(
        "Ask a question about the indexed research papers:",
        placeholder="e.g. What are reflection tokens in Self-RAG?"
    )

    if st.button("Ask ↗", type="primary") and question:
        with st.spinner("Retrieving relevant chunks..."):
            results = vectorstore.similarity_search(question, k=top_k)

        # Build context
        context = ""
        for i, doc in enumerate(results, 1):
            page = doc.metadata.get('page', 'N/A')
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            context += f"\n[Source {i} - {source} - Page {page}]:\n{doc.page_content}\n"

        # Generate answer
        with st.spinner("Generating answer..."):
            system_prompt = """You are an expert AI research assistant helping students and researchers understand academic papers.
Answer using ONLY the provided context from the indexed research papers.
Write in clear, well-structured paragraphs that are easy to understand.
Explain technical terms briefly when you use them.
Always cite which source and page supports each claim (e.g. Source 1 - Page 3).
If the question is asking about a person, place, event, or topic NOT directly discussed as a research concept in the papers, say exactly:
'This question is outside the scope of the indexed research papers.'
If the retrieved context only mentions the topic as a passing example or name, treat it as outside scope.
Never fabricate information or add details not present in the context."""

            user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a cited answer."
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = llm.invoke(messages)
            answer = response.content

        # Score
        with st.spinner("Evaluating answer quality..."):
            f_score = score_faithfulness(answer, context, llm)
            r_score = score_relevancy(question, answer, llm)

        # Display answer
        st.subheader("Answer")
        st.write(answer)

        # Display scores
        st.subheader("Evaluation Scores")
        m1, m2 = st.columns(2)
        m1.metric(
            "Faithfulness",
            f"{f_score:.2f}",
            help="How grounded is the answer in retrieved context? 1.0 = fully grounded"
        )
        m2.metric(
            "Answer Relevancy",
            f"{r_score:.2f}",
            help="How well does the answer address the question? 1.0 = fully relevant"
        )

        # Progress bars
        st.progress(f_score, text=f"Faithfulness: {f_score*100:.0f}%")
        st.progress(r_score, text=f"Answer Relevancy: {r_score*100:.0f}%")

        # Display retrieved chunks
        st.subheader(f"Retrieved Chunks (top {top_k})")
        for i, doc in enumerate(results, 1):
            page = doc.metadata.get('page', 'N/A')
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            with st.expander(f"Chunk {i} — {source} · Page {page}"):
                st.write(doc.page_content)

# ========== TAB 2: CONFIG COMPARISON ==========
with tab2:
    st.subheader("Configuration Comparison Results")
    st.caption("Evaluated across 10 test questions using LLM-as-judge faithfulness scoring")

    import pandas as pd
    config_data = {
        "Chunk Size": [250, 500, 1000],
        "# Chunks": [4314, 2066, 1086],
        "Faithfulness": ["67.1%", "65.7%", "68.6% ✓ best"],
        "Answer Relevancy": ["67.1%", "68.6%", "75.7% ✓ best"],
        "Precision@6": ["81.0%", "88.1% ✓ best", "83.3%"],
    }
    df = pd.DataFrame(config_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.info("""
**Key finding:** Larger chunk sizes (1000) improve faithfulness and answer relevancy
because more context is preserved per chunk. Smaller chunks (500) improve retrieval
precision because each chunk is more focused on a single idea.
This reveals a fundamental RAG tradeoff: retrieval precision vs generation quality.
    """)

    st.subheader("Retrieval Metrics (Before vs After Query Refinement)")
    retrieval_data = {
        "Metric": ["Precision@3", "Recall@3", "MRR"],
        "Before (original queries)": ["70.0%", "70.0%", "70.0%"],
        "After (refined queries)": ["90.0%", "90.0%", "90.0%"],
        "Improvement": ["+20%", "+20%", "+20%"],
    }
    df2 = pd.DataFrame(retrieval_data)
    st.dataframe(df2, use_container_width=True, hide_index=True)

# ========== TAB 3: SAFETY EVALUATION ==========
with tab3:
    st.subheader("Safety Evaluation Results")
    st.caption("Tests whether the system correctly refuses out-of-scope queries")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Out-of-scope questions (should refuse)**")
        out_of_scope = [
            ("What is the capital of France?", "✅ Refused"),
            ("Who is Elon Musk?", "✅ Refused"),
            ("Write me a poem about the ocean.", "✅ Refused"),
            ("What is the recipe for biryani?", "✅ Refused"),
            ("Who won the FIFA World Cup 2022?", "✅ Refused"),
            ("What is 2 + 2?", "✅ Refused"),
            ("History of the Roman Empire?", "✅ Refused"),
        ]
        for q, result in out_of_scope:
            st.markdown(f"{result} *{q}*")

    with col_b:
        st.markdown("**In-scope questions (should answer)**")
        in_scope = [
            ("What are reflection tokens in Self-RAG?", "✅ Answered"),
            ("What is chain of thought prompting?", "✅ Answered"),
            ("How does Mistral 7B differ from other LLMs?", "✅ Answered"),
        ]
        for q, result in in_scope:
            st.markdown(f"{result} *{q}*")

    st.divider()
    s1, s2 = st.columns(2)
    s1.metric("Out-of-scope refusal rate", "100%", "7/7 correctly refused")
    s2.metric("In-scope answer rate", "100%", "3/3 correctly answered")

    st.success("""
**Safety mechanism:** The system prompt explicitly instructs the LLM to answer
only from retrieved context and decline anything outside the indexed papers.
This prevents hallucination on out-of-scope queries without any model fine-tuning.
    """)