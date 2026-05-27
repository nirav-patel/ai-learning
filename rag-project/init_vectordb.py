#!/usr/bin/env python3
"""
Initialize ChromaDB Vector Database
Simple setup for storing and searching embeddings
"""

import chromadb
from sentence_transformers import SentenceTransformer

print("🗄️ Initializing ChromaDB Vector Database")
print("=" * 50)

# Initialize ChromaDB client (in-memory for simplicity)
print("1. Creating ChromaDB client...")
client = chromadb.Client()
print("   ✅ ChromaDB client created")

# Create a collection for our documents
print("2. Creating collection for TechCorp documents...")
collection = client.create_collection("techcorp_docs")
print("   ✅ Collection 'techcorp_docs' created")

# Load embedding model to show dimensions
print("3. Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"   ✅ Model loaded: {model.get_sentence_embedding_dimension()} dimensions")

# Test with a simple document
print("4. Testing with sample document...")
test_doc = "TechCorp allows remote work up to 3 days per week"
test_embedding = model.encode([test_doc])
print(f"   ✅ Sample embedding created: {len(test_embedding[0])} dimensions")

# Add test document to collection
collection.add(
    documents=[test_doc],
    ids=["test_doc_1"]
)
print("   ✅ Test document added to collection")

# Verify collection
print("5. Verifying collection...")
count = collection.count()
print(f"   ✅ Collection contains {count} documents")

print()
print("🎉 ChromaDB Vector Database Initialized Successfully!")
print(f"📊 Collection: techcorp_docs")
print(f"📊 Embedding dimensions: {model.get_sentence_embedding_dimension()}")
print(f"📊 Documents stored: {count}")

# Create completion marker
with open("vectordb_initialized.txt", "w") as f:
    f.write("ChromaDB vector database initialized successfully")

print("✅ Initialization complete!")
