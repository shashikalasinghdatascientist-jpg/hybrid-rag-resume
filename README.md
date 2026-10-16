# Hybrid Evidence-Based RAG System for Resume Analysis

This project implements a high-precision Retrieval-Augmented Generation (RAG) system designed to answer questions about a professional resume with 100% factual reliability. Unlike standard RAG systems, this implementation uses a **Verify-then-Answer** loop to eliminate hallucinations.

## 🚀 Key Features

- **Hybrid Retrieval**: Combines **Dense Vector Search** (semantic meaning) and **Sparse BM25 Search** (keyword matching) to ensure no critical detail is missed.
- **Cross-Encoder Reranking**: Uses a `cross-encoder/ms-marco-MiniLM-L-6-v2` model to re-score the top retrieval candidates, ensuring only the most relevant context is passed to the LLM.
- **Evidence-Based Generation**: Prompts Claude 3.5 Sonnet to provide verbatim quotes from the source document for every claim made.
- **Strict Verbatim Validation**: A custom validation layer that checks if the LLM's evidence exists exactly as a substring in the source PDF, flagging any "hallucinations" immediately.

## 🛠️ Technical Stack

- **LLM**: Claude 3.5 Sonnet (Anthropic API)
- **Framework**: LangChain
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Reranker**: Sentence-Transformers Cross-Encoder
- **PDF Parsing**: PyMuPDF (fitz)
- **Language**: Python 3.11+

## 📋 System Architecture

`PDF` $\rightarrow$ `PyMuPDF Parsing` $\rightarrow$ `Recursive Chunking` $\rightarrow$ `ChromaDB` $\rightarrow$ `Hybrid Search (Vector + BM25)` $\rightarrow$ `Cross-Encoder Reranking` $\rightarrow$ `Evidence-Based LLM Generation` $\rightarrow$ `Verbatim Validation` $\rightarrow$ `Final Answer`

## 🏃 Getting Started

### Prerequisites
- Python 3.11+
- GitHub CLI (`gh`)
- Anthropic API Key

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/shashikalasinghdatascientist-jpg/hybrid-rag-resume.git
   cd hybrid-rag-resume
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API key in a `.env` file:
   ```env
   ANTHROPIC_API_KEY=your_api_key_here
   ```

### Usage
1. **Ingest your resume:**
   Place your resume PDF in the project root and run:
   ```bash
   python3 ingest.py
   ```

2. **Query the system:**
   ```bash
   python3 main.py
   ```

## 🧪 Validation Example

**Question:** "How many years of experience does the candidate have?"
**LLM Answer:** "The candidate has 7+ years of experience."
**Evidence:** ["AI Engineer with 7+ years of experience"]
**Validation:** ✅ **Verified** (Quote found verbatim in source text).

---
Developed by [Shashikala Singh](https://github.com/shashikalasinghdatascientist-jpg)
