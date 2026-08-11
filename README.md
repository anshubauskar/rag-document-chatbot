# Document Q&A Chatbot (RAG-based Retrieval System)

A Retrieval-Augmented Generation (RAG) system that answers natural-language
questions using information retrieved directly from a document knowledge
base — instead of relying purely on a language model's internal memory.

## Problem Statement

Large Language Models don't know about private or domain-specific documents
(company policies, internal manuals, personal notes, etc.), and often
hallucinate answers when asked about topics outside their training data.
This project solves that by retrieving the most relevant document content
first, and using it to ground the model's answer — with the source visible.

## Architecture

```
Documents (.txt)
      |
      v
Chunking (line-level text splitting)
      |
      v
Embeddings (Sentence-Transformers: all-MiniLM-L6-v2)
      |
      v
Vector Index (FAISS - similarity search)
      |
      v
User Query --> Top-K Relevant Chunks
      |
      v
Local LLM (Ollama) --> Grounded Answer   [in progress]
      |
      v
Streamlit Web App                        [in progress]
```

## Tech Stack

Python, Sentence-Transformers, FAISS, Ollama (local LLM), Streamlit, NumPy

## Features

- Semantic chunking of source documents
- Vector embeddings using a pre-trained Sentence-Transformer model
- Fast similarity search over embeddings using FAISS
- Retrieves the most relevant document chunks for any natural-language query
- Local LLM answer generation and a Streamlit web UI (in progress)

## Project Structure

```
rag-document-chatbot/
├── documents/              # Sample knowledge-base documents
│   ├── hr_policy.txt
│   ├── leave_policy.txt
│   └── it_security_policy.txt
├── src/
│   └── rag_pipeline.py     # Chunking, embeddings, FAISS search
├── requirements.txt
└── README.md
```

## How to Run

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the retrieval pipeline
python src/rag_pipeline.py
```

## Sample Output

```
Total chunks: 9
Embedding shape: (9, 384)

Question: How many casual leaves do I get?
Top matching chunks:
- Employees are entitled to 12 paid casual leaves per year.
- Unused casual leaves cannot be carried forward to the next year.
```

## Current Status

- [x] Document loading and chunking
- [x] Embedding generation
- [x] FAISS vector search / retrieval
- [ ] Local LLM integration (Ollama) for answer generation
- [ ] Streamlit web app for interactive demo

## Future Improvements

- Support PDF/DOCX documents via `pdfplumber`
- Smarter chunking (sentence/paragraph-aware instead of line-based)
- Source citation in generated answers
- Swap sample documents for a larger, domain-specific knowledge base

## Skills Demonstrated

Vector embeddings, semantic search, FAISS indexing, RAG architecture,
Python modular code design, local/offline LLM integration.
