# TrustRAGBench

A research-grade Retrieval-Augmented Generation (RAG) system for academic paper Q&A. Includes retrieval evaluation, hallucination/faithfulness evaluation, and an interactive dashboard.

## 🚀 Live Demo

**[https://trustragbench.streamlit.app/](https://trustragbench.streamlit.app/)**

## What It Does

- Ask questions about 10 AI/ML research papers and get cited, grounded answers
- Answers are generated using LLaMA 3.1 8B (via Groq API) — never from the model's own memory
- Every answer includes page citations from the source paper
- System refuses out-of-scope questions rather than hallucinating

## Evaluation Results

| Metric | Result |
|---|---|
| Precision@3 | 90.0% |
| Recall@3 | 90.0% |
| MRR | 90.0% |
| Faithfulness (chunk=1000) | 68.6% |
| Answer Relevancy (chunk=1000) | 75.7% |
| Safety refusal rate | 100% (7/7) |

## Papers Indexed

| Paper | Authors | Year |
|---|---|---|
| Self-RAG | Asai et al. | 2023 |
| RAG Original | Lewis et al. | 2020 |
| CRAG | Yan et al. | 2024 |
| HyDE | Gao et al. | 2022 |
| LLaMA 2 | Touvron et al. | 2023 |
| Mistral 7B | Jiang et al. | 2023 |
| Chain-of-Thought | Wei et al. | 2022 |
| RAGAS | Es et al. | 2023 |
| Sentence-BERT | Reimers et al. | 2019 |
| ReAct | Yao et al. | 2022 |

## Tech Stack

- **RAG Framework:** LangChain
- **Vector Database:** ChromaDB
- **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2
- **LLM:** LLaMA 3.1 8B via Groq API
- **PDF Parsing:** PyPDF
- **Dashboard:** Streamlit
- **Version Control:** Git/GitHub

## Project Structure
TrustRAGBench/
├── data/ # Indexed research papers (10 PDFs)
├── chroma_db/ # Persisted vector store (chunk=500)
├── chroma_db_chunk250/ # Vector store (chunk=250)
├── chroma_db_chunk1000/ # Vector store (chunk=1000)
├── app.py # Streamlit dashboard
├── rag_pipeline.py # Baseline RAG pipeline
├── rag_with_llm.py # RAG + LLM answer generation
├── evaluation.py # Retrieval evaluation (Precision@k, Recall@k, MRR)
├── ragas_eval.py # Faithfulness evaluation (LLM-as-judge)
├── config_comparison.py # Multi-config comparison
├── safety_eval.py # Safety evaluation
├── requirements.txt # Python dependencies
└── .gitignore