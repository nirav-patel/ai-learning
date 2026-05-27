#!/usr/bin/env python3
"""
Simple BM25 Search Demo
"""

from rank_bm25 import BM25Okapi
import re
from utils import get_doc_info

print("🔍 BM25 Search Demo")
print("=" * 50)

# Load documents from techcorp-docs
docs, doc_paths = get_doc_info()
print(f"📚 Loaded {len(docs)} documents\n")

# Tokenize documents
tokenized_docs = [re.sub(r'[^a-zA-Z\s]', '', doc.lower()).split() for doc in docs]

# Create BM25 index
bm25 = BM25Okapi(tokenized_docs)

# Example searches
queries = ["remote work policy", "health insurance benefits", "pet policy dogs"]

for query in queries:
    print(f"🔎 Searching for: '{query}'")
    
    # Tokenize query
    tokenized_query = re.sub(r'[^a-zA-Z\s]', '', query.lower()).split()
    
    # Get BM25 scores
    scores = bm25.get_scores(tokenized_query)
    
    # Get top results
    top_indices = scores.argsort()[-3:][::-1]
    
    print("Results:")
    for i, idx in enumerate(top_indices, 1):
        # Show only document path and score
        doc_name = doc_paths[idx].split('/')[-1]  # Just the filename
        print(f"  {i}. Score: {scores[idx]:.4f} - {doc_name}")
    print()

print("✅ BM25 search completed!")