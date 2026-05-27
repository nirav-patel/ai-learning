#!/usr/bin/env python3
"""
Save Vector Database to File
Demonstrate file persistence for ChromaDB
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
import os

print("💾 Saving Vector Database to File")
print("=" * 50)

# Initialize ChromaDB and model
print("1. Setting up vector database...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("techcorp_docs")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("   ✅ Persistent ChromaDB and model ready")

# Add sample documents
print("2. Adding sample documents...")
sample_docs = [
    "TechCorp allows remote work up to 3 days per week",
    "Employees can bring pets to work on Fridays",
    "Company provides health insurance and dental coverage",
    "Remote workers must use approved equipment"
]

collection.add(
    documents=sample_docs,
    ids=[f"doc_{i+1}" for i in range(len(sample_docs))]
)
print(f"   ✅ Added {len(sample_docs)} documents")

# Save collection data to file
print("3. Saving to file...")
collection_data = {
    "documents": sample_docs,
    "ids": [f"doc_{i+1}" for i in range(len(sample_docs))],
    "count": len(sample_docs)
}

# Save as JSON file
with open("vectordb_backup.json", "w") as f:
    json.dump(collection_data, f, indent=2)

print("   ✅ Saved to vectordb_backup.json")

# Verify file was created
if os.path.exists("vectordb_backup.json"):
    file_size = os.path.getsize("vectordb_backup.json")
    print(f"   ✅ File size: {file_size} bytes")

print()
print("💡 File Persistence Benefits:")
print("✅ Data survives system restarts")
print("✅ Can be shared between applications")
print("✅ Backup and restore capabilities")
print("✅ Version control for document changes")

print()
print("🎉 Vector Database Saved Successfully!")
print(f"📊 Documents saved: {len(sample_docs)}")
print(f"📊 File: vectordb_backup.json")
print(f"📊 File size: {file_size} bytes")

# Create completion marker
with open("vectordb_saved.txt", "w") as f:
    f.write("Vector database saved to file successfully")

print("✅ File persistence complete!")
