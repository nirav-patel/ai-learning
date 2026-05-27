#!/usr/bin/env python3
"""
Keyword Search Limitations Demo
Shows why keyword search fails for semantic queries
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import read_techcorp_docs

print("🔍 Keyword Search Limitations Demo")
print("=" * 50)

# Load documents (without verbose output)
docs, doc_paths = read_techcorp_docs()

# Create TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs)

# Test query that demonstrates limitations
query = "distributed workforce policies"
print(f"🔎 Searching for: '{query}'")

# Transform query to TF-IDF
query_vector = vectorizer.transform([query])

# Calculate similarities
similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

# Get top results
top_indices = similarities.argsort()[-3:][::-1]

print("Results:")
for i, idx in enumerate(top_indices, 1):
    doc_name = doc_paths[idx].split('/')[-1]
    print(f"  {i}. Score: {similarities[idx]:.4f} - {doc_name}")

# Check if we found relevant documents
if similarities[top_indices[0]] < 0.05:
    print("  ❌ No relevant documents found!")
else:
    print("  ✅ Found some matches")

print("\n💡 Problem: 'distributed workforce policies' doesn't match 'remote work policy'")
print("We need semantic search that understands meaning!")

print("\n✅ Keyword limitation demo completed!")
