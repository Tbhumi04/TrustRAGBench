"""
ragas_eval.py
Custom faithfulness and answer relevancy evaluation.
Faithfulness: does the answer stay grounded in retrieved context?
Answer Relevancy: does the answer actually address the question?
"""

import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ---------- LOAD VECTOR STORE ----------
print("Loading vector store...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)
print("Vector store loaded.\n")

# ---------- SETUP LLM ----------
llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.0
)

# ---------- RAG FUNCTION ----------
def get_rag_answer(question):
    results = vectorstore.similarity_search(question, k=6)
    context = ""
    contexts = []
    for i, doc in enumerate(results, 1):
        page = doc.metadata.get('page', 'N/A')
        context += f"\n[Source {i} - Page {page}]:\n{doc.page_content}\n"
        contexts.append(doc.page_content)

    system_prompt = """You are a research assistant. Answer using ONLY the provided context.
Always cite which source supports your answer.
If context is insufficient, say: 'The available documents do not contain sufficient information.'
Never fabricate information."""

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a cited answer."
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = llm.invoke(messages)
    return response.content, context

# ---------- FAITHFULNESS SCORER ----------
def score_faithfulness(question, answer, context):
    """Ask LLM to judge if answer is grounded in context. Returns score 0.0-1.0"""
    prompt = f"""You are an evaluation judge. 

Given this context:
{context[:2000]}

And this answer:
{answer}

Judge: Is every claim in the answer supported by the context above?
Reply with ONLY a number between 0.0 and 1.0 where:
1.0 = fully grounded, every claim supported by context
0.5 = partially grounded, some claims not in context  
0.0 = not grounded, answer contains fabricated information

Reply with ONLY the number, nothing else."""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        score = float(response.content.strip())
        return min(max(score, 0.0), 1.0)
    except:
        return 0.5

# ---------- RELEVANCY SCORER ----------
def score_relevancy(question, answer):
    """Ask LLM to judge if answer actually addresses the question. Returns score 0.0-1.0"""
    prompt = f"""You are an evaluation judge.

Question: {question}
Answer: {answer}

Judge: How well does this answer address the question?
Reply with ONLY a number between 0.0 and 1.0 where:
1.0 = directly and completely answers the question
0.5 = partially answers the question
0.0 = does not answer the question at all

Reply with ONLY the number, nothing else."""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        score = float(response.content.strip())
        return min(max(score, 0.0), 1.0)
    except:
        return 0.5

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

# ---------- RUN EVALUATION ----------
print("Running faithfulness and relevancy evaluation...\n")
print("="*70)

faithfulness_scores = []
relevancy_scores = []

for q in test_questions:
    print(f"Q: {q[:60]}...")
    answer, context = get_rag_answer(q)
    f_score = score_faithfulness(q, answer, context)
    r_score = score_relevancy(q, answer)
    faithfulness_scores.append(f_score)
    relevancy_scores.append(r_score)
    print(f"  Faithfulness: {f_score:.2f} | Answer Relevancy: {r_score:.2f}")
    print()

# ---------- SUMMARY ----------
avg_faith = sum(faithfulness_scores) / len(faithfulness_scores)
avg_rel = sum(relevancy_scores) / len(relevancy_scores)

print("="*70)
print("SUMMARY")
print("="*70)
print(f"Average Faithfulness:     {avg_faith:.4f}  ({avg_faith*100:.1f}%)")
print(f"Average Answer Relevancy: {avg_rel:.4f}  ({avg_rel*100:.1f}%)")
print("="*70)
print("\nNote: Faithfulness measures if answers stay grounded in retrieved context.")
print("Note: Answer Relevancy measures if answers actually address the question.")