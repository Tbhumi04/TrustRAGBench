"""
evaluation.py
Retrieval evaluation script for TrustRAGBench.
Computes Precision@k, Recall@k, and MRR across a test question set.
"""

import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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

# ---------- TEST SET ----------
# Each entry: question + list of keywords that MUST appear in a relevant chunk
# This is our ground truth — what a correct retrieved chunk should contain
test_set = [
    {
        "question": "What are reflection tokens in Self-RAG?",
        "relevant_keywords": ["reflection token", "critique", "retrieval token"]
    },
    {
        "question": "What is HyDE and how does it improve retrieval?",
        "relevant_keywords": ["hypothetical", "hyde", "document embeddings"]
    },
    {
        "question": "What is chain of thought prompting?",
        "relevant_keywords": ["chain of thought", "reasoning", "step by step"]
    },
    {
        "question": "What architecture does Mistral 7B use?",
        "relevant_keywords": ["sliding window", "grouped query", "mistral"]
    },
    {
        "question": "How does ReAct combine reasoning and acting?",
        "relevant_keywords": ["react", "reasoning", "acting", "action"]
    },
    {
        "question": "How does Sentence-BERT generate sentence embeddings?",
        "relevant_keywords": ["siamese", "sentence embedding", "bert"]
    },
    {
        "question": "How does RAG retrieve documents to answer questions?",
        "relevant_keywords": ["retrieval", "retrieve", "generator", "passage"]
    },
    {
        "question": "How does CRAG use a retrieval evaluator to fix bad retrievals?",
        "relevant_keywords": ["retrieval evaluator", "corrective", "confidence", "web search"]
    },
    {
        "question": "What safety techniques were used in LLaMA 2 chat models?",
        "relevant_keywords": ["safety", "rlhf", "fine-tuning", "chat", "red team"]
    },
    {
    "question": "How does RAG combine retrieval with language generation?",
    "relevant_keywords": ["retriever", "generator", "retrieved", "generation", "passages"]
},
]

# ---------- EVALUATION FUNCTIONS ----------
def is_relevant(chunk_text, keywords):
    """Check if a chunk contains any of the expected keywords."""
    text_lower = chunk_text.lower()
    return any(kw.lower() in text_lower for kw in keywords)

def evaluate(k=3):
    print(f"Running evaluation with k={k}")
    print("="*60)

    precisions = []
    recalls = []
    mrr_scores = []

    for item in test_set:
        question = item["question"]
        keywords = item["relevant_keywords"]

        # Retrieve top-k chunks
        results = vectorstore.similarity_search(question, k=k)

        # Check which retrieved chunks are relevant
        relevant_flags = [is_relevant(doc.page_content, keywords) for doc in results]
        num_relevant_retrieved = sum(relevant_flags)

        # Precision@k: how many of the k retrieved chunks are relevant
        precision = num_relevant_retrieved / k

        # Recall@k: assuming ground truth has at least 1 relevant chunk
        recall = 1.0 if num_relevant_retrieved > 0 else 0.0

        # MRR: reciprocal rank of first relevant result
        mrr = 0.0
        for rank, flag in enumerate(relevant_flags, 1):
            if flag:
                mrr = 1.0 / rank
                break

        precisions.append(precision)
        recalls.append(recall)
        mrr_scores.append(mrr)

        # Print per-question result
        status = "✓" if num_relevant_retrieved > 0 else "✗"
        print(f"{status} Q: {question[:55]}...")
        print(f"  Precision@{k}: {precision:.2f} | Recall@{k}: {recall:.2f} | MRR: {mrr:.2f}")
        print()

    # Overall averages
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_mrr = sum(mrr_scores) / len(mrr_scores)

    print("="*60)
    print("OVERALL RESULTS")
    print("="*60)
    print(f"Average Precision@{k}: {avg_precision:.4f}  ({avg_precision*100:.1f}%)")
    print(f"Average Recall@{k}:    {avg_recall:.4f}  ({avg_recall*100:.1f}%)")
    print(f"Average MRR:           {avg_mrr:.4f}  ({avg_mrr*100:.1f}%)")
    print("="*60)

    return avg_precision, avg_recall, avg_mrr

# ---------- RUN FOR DIFFERENT K VALUES ----------
print("\n--- Evaluation at k=3 ---")
evaluate(k=3)

print("\n--- Evaluation at k=6 ---")
evaluate(k=6)