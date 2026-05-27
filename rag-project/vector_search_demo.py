#!/usr/bin/env python3
"""
Vector Search Demo
Demonstrate semantic search using ChromaDB
"""

import chromadb
from sentence_transformers import SentenceTransformer

print("🔍 Vector Search Demo")
print("=" * 40)

# Initialize ChromaDB and model
client = chromadb.Client()
collection = client.create_collection("techcorp_docs")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Add sample documents
sample_docs = [
    "TechCorp allows remote work up to 3 days per week with manager approval",
    "Employees can bring their pets to work on Fridays",
    "The company provides health insurance and dental coverage",
    "Remote workers must use company-approved equipment and software"
]

collection.add(
    documents=sample_docs,
    ids=[f"sample_{i+1}" for i in range(len(sample_docs))]
)

# Test vector search
query = "Can I work from home?"
results = collection.query(
    query_texts=[query],
    n_results=2
)

print(f"Query: '{query}'")
for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
    similarity = 1 - distance
    print(f"  {i+1}. Similarity: {similarity:.3f} - {doc}")

print("\n✅ Vector search demo completed!")
