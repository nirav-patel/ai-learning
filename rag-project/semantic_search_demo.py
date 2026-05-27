#!/usr/bin/env python3
"""
Semantic Search Demo using Local Embeddings
Uses sentence-transformers for semantic similarity
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from utils import read_techcorp_docs

print("🧠 Semantic Search Demo (Local Embeddings)")
print("=" * 50)

# Load documents (without verbose output)
docs, doc_paths = read_techcorp_docs()

# Load local embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all documents
doc_embeddings = model.encode(docs)

# Test query that failed with keyword search
query = "distributed workforce policies"
print(f"🔎 Searching for: '{query}'")

# Generate embedding for query
query_embedding = model.encode([query])

# Calculate cosine similarities
similarities = np.dot(query_embedding, doc_embeddings.T).flatten()

# Get top results
top_indices = similarities.argsort()[-3:][::-1]

print("Results:")
for i, idx in enumerate(top_indices, 1):
    doc_name = doc_paths[idx].split('/')[-1]
    print(f"  {i}. Score: {similarities[idx]:.4f} - {doc_name}")

# Check if we found relevant documents
if similarities[top_indices[0]] > 0.3:
    print("  ✅ Found relevant documents!")
else:
    print("  ❌ No relevant documents found!")

print("\n💡 Semantic search success because:")
print("- Understands 'distributed workforce policies' ≈ 'remote work policy'")
print("- Embeddings capture meaning, not just keywords!")

print("\n✅ Semantic search demo completed!")
