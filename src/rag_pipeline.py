"""
RAG Pipeline - Core Module
Handles: loading documents, chunking, embeddings, and FAISS vector search.

This module currently implements the RETRIEVAL half of the RAG system
(Steps 1-3). LLM answer generation (Step 4) will be added in generate.py.
"""

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_chunks(folder_path):
    """Read every .txt file in the folder and split it into line-level chunks."""
    chunks = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    chunks.append(line)
    return chunks


def create_embeddings(chunks, model):
    """Convert text chunks into numeric vector embeddings."""
    embeddings = model.encode(chunks)
    return embeddings


def build_index(embeddings):
    """Build a FAISS index for fast similarity search over the embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index


def search(query, model, index, chunks, top_k=2):
    """Find the top_k most relevant chunks for a given user query."""
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [chunks[i] for i in indices[0]]
    return results


if __name__ == "__main__":
    # Quick manual test of the retrieval pipeline
    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunks = load_chunks("documents")
    embeddings = create_embeddings(chunks, model)
    index = build_index(embeddings)

    print("Total chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)

    query = "How many casual leaves do I get?"
    top_chunks = search(query, model, index, chunks)

    print("\nQuestion:", query)
    print("Top matching chunks:")
    for chunk in top_chunks:
        print("-", chunk)
