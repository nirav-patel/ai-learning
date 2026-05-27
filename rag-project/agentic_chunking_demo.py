#!/usr/bin/env python3
"""
Agentic Chunking Demo
Using LLM to intelligently split documents based on semantic meaning

This script demonstrates **Agentic Chunking** - the most advanced chunking method
where an AI model analyzes the document and decides optimal split points based on
topic shifts and semantic coherence, rather than arbitrary character counts.
"""
import os
import sys
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("🤖 Agentic Chunking Demo")
print("=" * 50)

# Configuration - using environment variables for API access
API_KEY = os.environ.get("OPENAI_API_KEY")
API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
MODEL_NAME = "openai/gpt-4.1-mini"

if not API_KEY:
    print("❌ Error: OPENAI_API_KEY not found.")
    print("Please ensure the environment is configured correctly.")
    sys.exit(1)

print(f"🔌 API Endpoint: {API_BASE}")
print(f"🧠 Model: {MODEL_NAME}")
print()

# Sample document with multiple distinct topics
sample_document = """
TechCorp Company Overview

Company History: Founded in 1995 in a garage in Silicon Valley, TechCorp started as a small software consultancy. By 2000, it had grown to 500 employees and went public. The early years were marked by rapid expansion and the release of its flagship product, the TechOS. The founders, Jane Smith and John Doe, built the company on principles of innovation and customer focus.

Product Lineup: Today, TechCorp offers a wide range of enterprise software solutions. The CloudSuite is our most popular offering, providing scalable cloud infrastructure for businesses of all sizes. We also offer DataGuard for enterprise security, protecting sensitive data with military-grade encryption. AI-Core handles machine learning integration, making AI accessible to non-technical teams. Each product is designed to work seamlessly with the others.

Remote Work Policy: Employees may work remotely up to 3 days per week with manager approval. Remote work must be conducted using company-approved devices with VPN access enabled. All employees must be available during core hours (10 AM - 4 PM) and maintain regular communication with their team. Remote work is not a substitute for childcare or eldercare.

Future Vision: Looking ahead, TechCorp is betting big on quantum computing. We plan to invest $1B over the next 5 years in R&D for quantum technologies. Our goal is to be the first company to offer commercial quantum cloud services by 2030. This investment will create new positions for quantum researchers and engineers across all our locations.
"""

print("📄 Sample Document:")
print(f"Length: {len(sample_document)} characters")
print(f"Contains 4 distinct topics: History, Products, Remote Work, Future")
print()

# First, let's compare with basic chunking
print("🔧 Comparison: Basic Chunking vs Agentic Chunking")
print("-" * 50)

# Basic character-based chunking
basic_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

basic_chunks = basic_splitter.split_text(sample_document)
print(f"\n📊 Basic Chunking Result: {len(basic_chunks)} chunks")
print("   (Based on character count, may split mid-topic)")
for i, chunk in enumerate(basic_chunks, 1):
    preview = chunk[:60].replace('\n', ' ').strip()
    print(f"   Chunk {i}: {preview}...")
print()

# Agentic chunking using LLM
def agentic_chunking(text):
    """
    Uses an LLM to split text into semantically distinct chunks.
    The AI analyzes topic shifts and creates meaningful boundaries.
    """
    print("🤔 Agent is analyzing the document for semantic topic shifts...")
    
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=API_KEY,
        openai_api_base=API_BASE,
        temperature=0  # Deterministic output for consistency
    )

    # The Prompt: Instruct the LLM to act as a "Chunking Agent"
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert document editor specializing in semantic document analysis.
Your task is to split the provided text into semantically distinct chunks based on topic shifts.

Rules:
1. Keep related sentences together - don't break up a single topic
2. Split ONLY when the topic changes significantly (e.g., History -> Products -> Policy -> Future)
3. Each chunk should be about ONE coherent topic
4. Output the chunks separated by '---SPLIT---'
5. Do not modify the original text - just split it at appropriate boundaries
6. Include section headers with their content in the same chunk"""),
        ("user", "{text}")
    ])

    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({"text": text})
        # Split the response by our delimiter and clean up
        chunks = [c.strip() for c in response.split("---SPLIT---") if c.strip()]
        return chunks
    except Exception as e:
        print(f"\n❌ API Error: {e}")
        return []

# Run agentic chunking
agentic_chunks = agentic_chunking(sample_document)

if agentic_chunks:
    print(f"\n📊 Agentic Chunking Result: {len(agentic_chunks)} chunks")
    print("   (Based on semantic meaning and topic shifts)")
    print()
    
    for i, chunk in enumerate(agentic_chunks, 1):
        # Identify the likely topic from the chunk
        if "History" in chunk or "Founded" in chunk:
            topic = "Company History"
        elif "Product" in chunk or "CloudSuite" in chunk:
            topic = "Products"
        elif "Remote" in chunk or "work" in chunk.lower():
            topic = "Remote Work Policy"
        elif "Future" in chunk or "quantum" in chunk.lower():
            topic = "Future Vision"
        else:
            topic = "General"
        
        print(f"📦 Chunk {i} - Topic: {topic}")
        print(f"   Length: {len(chunk)} characters")
        preview = chunk[:80].replace('\n', ' ').strip()
        print(f"   Preview: {preview}...")
        print()

    # Comparison summary
    print("🔍 Comparison Summary:")
    print("-" * 50)
    print(f"Basic Chunking:   {len(basic_chunks)} chunks (character-based)")
    print(f"Agentic Chunking: {len(agentic_chunks)} chunks (semantic-based)")
    print()
    print("💡 Key Differences:")
    print("✅ Agentic chunking identifies natural topic boundaries")
    print("✅ Each chunk contains ONE coherent topic")
    print("✅ Better semantic coherence for RAG retrieval")
    print("✅ AI understands context and meaning")
    print("✅ No arbitrary character limit splitting")
    print()
    
    print("💡 When to Use Agentic Chunking:")
    print("✅ Documents with clear topic sections")
    print("✅ When semantic coherence is critical")
    print("✅ Complex documents with mixed content")
    print("✅ When retrieval quality matters more than speed")
    print()
    
    print("⚠️  Considerations:")
    print("• Requires LLM API calls (cost and latency)")
    print("• Best for smaller documents or preprocessing")
    print("• May need fallback for very large documents")
    
    # Create completion marker
    with open("agentic_chunking_complete.txt", "w") as f:
        f.write("Agentic chunking demo completed successfully")
    
    print("\n✅ Agentic chunking demo completed!")
else:
    print("\n⚠️ Agent failed to produce chunks. Check API connection.")
