#!/usr/bin/env python3
"""
Sentence-Aware Chunking Demo
Using spaCy for better sentence boundary detection
"""

from langchain_text_splitters import SpacyTextSplitter
import spacy

print("✂️ Sentence-Aware Chunking Demo")
print("=" * 50)

# Download spaCy model if not already present
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded successfully")
except OSError:
    print("⚠️  spaCy model not found. Using basic chunking instead.")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    nlp = None

# Sample document with complex sentences
sample_document = """
TechCorp Security Policy and Remote Work Guidelines

Employees working remotely must follow strict security protocols to protect company data and systems. All remote work must be conducted using company-approved devices and software, including laptops, monitors, and security software. Personal devices, including smartphones and tablets, are strictly prohibited for accessing company systems or storing confidential information.

The company provides VPN access to all remote employees, which must be used whenever accessing internal systems or databases. VPN connections must be established before accessing any company resources, and employees must ensure their internet connection is secure and private. Public Wi-Fi networks, including those in coffee shops, airports, and hotels, are not permitted for company work due to security risks.

All confidential documents must be stored in approved cloud storage systems with proper encryption and access controls. Local storage of sensitive information on personal computers or external drives is strictly forbidden. Employees must use strong passwords and enable two-factor authentication for all company accounts and systems.

Regular security training sessions are mandatory for all remote workers, covering topics such as phishing prevention, password management, and data handling procedures. Employees must complete these training modules within 30 days of starting remote work and annually thereafter. Failure to complete security training may result in suspension of remote work privileges.

Incident reporting procedures require immediate notification of any security breaches, suspicious activities, or potential data exposures to the IT security team. Employees must report incidents within 2 hours of discovery using the designated security hotline or email system. Delayed reporting may result in disciplinary action and potential legal consequences.
"""

print("📄 Sample Document:")
print(f"Length: {len(sample_document)} characters")
print(f"Complex sentences with multiple clauses")
print()

# Test 1: Basic character-based chunking
print("🔧 Test 1: Basic Character-Based Chunking")
print("-" * 50)

from langchain_text_splitters import RecursiveCharacterTextSplitter

basic_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

basic_chunks = basic_splitter.split_text(sample_document)

print(f"Created {len(basic_chunks)} chunks:")
for i, chunk in enumerate(basic_chunks, 1):
    print(f"Chunk {i}: {chunk[:100]}...")
    # Check if chunk breaks mid-sentence
    if not chunk.strip().endswith(('.', '!', '?')):
        print("  ⚠️  Breaks mid-sentence!")
    else:
        print("  ✅ Ends at sentence boundary")
    print()

# Test 2: Sentence-aware chunking (if spaCy available)
if nlp:
    print("🔧 Test 2: Sentence-Aware Chunking with spaCy")
    print("-" * 50)
    
    sentence_splitter = SpacyTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    
    sentence_chunks = sentence_splitter.split_text(sample_document)
    
    print(f"Created {len(sentence_chunks)} chunks:")
    for i, chunk in enumerate(sentence_chunks, 1):
        print(f"Chunk {i}: {chunk[:100]}...")
        # Check if chunk breaks mid-sentence
        if not chunk.strip().endswith(('.', '!', '?')):
            print("  ⚠️  Breaks mid-sentence!")
        else:
            print("  ✅ Ends at sentence boundary")
        print()
    
    print("🔍 Comparison:")
    print("Basic chunking:")
    print("  - May break mid-sentence")
    print("  - Can lose semantic meaning")
    print("  - Simpler implementation")
    print()
    print("Sentence-aware chunking:")
    print("  - Preserves sentence boundaries")
    print("  - Better semantic coherence")
    print("  - More natural chunk breaks")
    print("  - Better for NLP processing")
else:
    print("⚠️  spaCy not available - skipping sentence-aware chunking demo")
    print("💡 Install spaCy with: python -m spacy download en_core_web_sm")

print("💡 Sentence Boundary Benefits:")
print("✅ Preserves complete thoughts and ideas")
print("✅ Better semantic coherence")
print("✅ More natural chunk breaks")
print("✅ Improved readability")
print("✅ Better for language processing")

# Create completion marker
with open("sentence_chunking_complete.txt", "w") as f:
    f.write("Sentence chunking demo completed successfully")

print("\n✅ Sentence chunking demo completed!")
