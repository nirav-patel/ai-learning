#!/usr/bin/env python3
"""
Chunked Vector Search Demo
Compare search performance with and without chunking
"""

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("🔍 Chunked Vector Search Demo")
print("=" * 50)

# Initialize ChromaDB and model
client = chromadb.Client()
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample policy document
policy_document = """
TechCorp Remote Work Policy

Section 1: Eligibility and Approval
Employees may work remotely up to 3 days per week with manager approval. 
Remote work days must be scheduled in advance and approved by your direct supervisor.
All remote work must comply with company security policies and use approved equipment.

Section 2: Equipment Requirements
Remote employees must have a secure and reliable internet connection with minimum speeds of 25 Mbps download and 5 Mbps upload.
All work must be performed on company-approved devices and software.
Employees must use VPN when accessing company systems.
Personal devices are not permitted for work purposes.

Section 3: Workspace Standards
Remote work is not a substitute for childcare or eldercare responsibilities.
Employees must have a dedicated workspace free from distractions.
The workspace must be professional and suitable for video calls.
Background noise should be minimized during meetings.

Section 4: Communication Requirements
Employees must be available during core business hours (9 AM - 5 PM local time).
Regular check-ins with managers are required.
Team meetings must be attended via video conference.
Email and instant messaging should be checked regularly.

Section 5: Security and Compliance
All company data must be handled according to security policies.
Confidential information must not be discussed in public spaces.
Documents must be stored in approved cloud systems only.
Regular security training must be completed.
"""

print("📄 Sample Policy Document:")
print(f"Length: {len(policy_document)} characters")
print()

# Test 1: Search WITHOUT chunking
print("🔧 Test 1: Search WITHOUT Chunking")
print("-" * 40)

# Create collection for non-chunked search
collection_no_chunking = client.create_collection("no_chunking")

# Store entire document as single chunk
collection_no_chunking.add(
    documents=[policy_document],
    ids=["full_document"]
)

print("Stored entire document as single chunk")
print()

# Test 2: Search WITH chunking
print("🔧 Test 2: Search WITH Chunking")
print("-" * 40)

# Create collection for chunked search
collection_chunked = client.create_collection("chunked")

# Split document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_text(policy_document)
print(f"Split document into {len(chunks)} chunks")

# Store chunks in vector database
chunk_ids = [f"chunk_{i+1}" for i in range(len(chunks))]
collection_chunked.add(
    documents=chunks,
    ids=chunk_ids
)

print("Stored chunks in vector database")
print()

# Test queries
test_queries = [
    "What are the internet speed requirements?",
    "Can I use my personal laptop for work?",
    "What are the workspace requirements?",
    "How often do I need to check in with my manager?"
]

print("🔍 Search Performance Comparison:")
print("=" * 50)

for query in test_queries:
    print(f"\nQuery: '{query}'")
    print("-" * 30)
    
    # Search without chunking
    results_no_chunking = collection_no_chunking.query(
        query_texts=[query],
        n_results=1
    )
    
    # Search with chunking
    results_chunked = collection_chunked.query(
        query_texts=[query],
        n_results=2
    )
    
    print("Without Chunking:")
    print(f"  Similarity: {1 - results_no_chunking['distances'][0][0]:.3f}")
    print(f"  Result: {results_no_chunking['documents'][0][0][:100]}...")
    print(f"  Problem: Returns entire document!")
    
    print("\nWith Chunking:")
    for i, (doc, distance) in enumerate(zip(results_chunked['documents'][0], results_chunked['distances'][0])):
        similarity = 1 - distance
        print(f"  Chunk {i+1} - Similarity: {similarity:.3f}")
        print(f"  Result: {doc[:100]}...")
        print(f"  Benefit: Focused, relevant information!")

print("\n💡 Chunking Benefits for Search:")
print("✅ More precise and relevant results")
print("✅ Focused information instead of entire documents")
print("✅ Better similarity scores for specific topics")
print("✅ Easier to find specific information")
print("✅ Improved user experience")
print("✅ Better context for LLM generation")

print("\n📊 Performance Summary:")
print(f"Without chunking: 1 large document, hard to find specific info")
print(f"With chunking: {len(chunks)} focused chunks, precise results")

# Create completion marker
with open("chunked_search_complete.txt", "w") as f:
    f.write("Chunked search demo completed successfully")

print("\n✅ Chunked search demo completed!")
