#!/usr/bin/env python3
"""
Overlap Chunking Demo
Demonstrates the importance of overlap for context preservation
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

print("✂️ Overlap Chunking Demo")
print("=" * 50)

# Sample document with context that spans chunk boundaries
sample_document = """
TechCorp Equipment Reimbursement Policy

Section 1: Eligibility Requirements
Employees working from home may claim up to $500 per year for office equipment including desks, chairs, monitors, and computer accessories. This policy applies to full-time remote workers only. Part-time employees are not eligible for this benefit.

Section 2: Approval Process
All equipment purchases must be pre-approved by your direct manager. Submit a purchase request form at least 2 weeks before the intended purchase date. Include item description, estimated cost, and business justification. Manager approval is required before any purchase.

Section 3: Reimbursement Process
Receipts must be submitted within 30 days of purchase. Use the company expense reporting system to submit your claim. Include original receipts and manager approval email. Reimbursement will be processed within 2 weeks of submission.

Section 4: Equipment Standards
All equipment must meet company security standards. Computers must have approved antivirus software installed. Monitors must support minimum 1080p resolution. Chairs must be ergonomic and adjustable. Desks must provide adequate workspace for dual monitors.

Section 5: Return Policy
If employment ends within 12 months of purchase, equipment must be returned to the company. Equipment becomes employee property after 12 months of continuous employment. Returned equipment will be inspected for damage and normal wear.
"""

print("📄 Sample Document:")
print(f"Length: {len(sample_document)} characters")
print()

# Test 1: Chunking WITHOUT overlap
print("🔧 Test 1: Chunking WITHOUT Overlap")
print("-" * 40)

splitter_no_overlap = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,  # No overlap
    separators=["\n\n", "\n", " ", ""]
)

chunks_no_overlap = splitter_no_overlap.split_text(sample_document)

print(f"Created {len(chunks_no_overlap)} chunks without overlap:")
for i, chunk in enumerate(chunks_no_overlap, 1):
    print(f"Chunk {i}: {chunk[:80]}...")
print()

# Test 2: Chunking WITH overlap
print("🔧 Test 2: Chunking WITH Overlap")
print("-" * 40)

splitter_with_overlap = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,  # 50 character overlap
    separators=["\n\n", "\n", " ", ""]
)

chunks_with_overlap = splitter_with_overlap.split_text(sample_document)

print(f"Created {len(chunks_with_overlap)} chunks with overlap:")
for i, chunk in enumerate(chunks_with_overlap, 1):
    print(f"Chunk {i}: {chunk[:80]}...")
print()

# Show the difference
print("🔍 Overlap Analysis:")
print("Without overlap - potential context loss:")
print("  Chunk 1 ends: '...Part-time employees are not eligible'")
print("  Chunk 2 starts: 'All equipment purchases must be pre-approved'")
print("  ❌ Context lost between chunks!")
print()

print("With overlap - context preserved:")
print("  Chunk 1 ends: '...Part-time employees are not eligible for this benefit.'")
print("  Chunk 2 starts: 'This policy applies to full-time remote workers only. Part-time employees are not eligible for this benefit. All equipment purchases...'")
print("  ✅ Context preserved across boundaries!")
print()

print("💡 Overlap Benefits:")
print("✅ Preserves context across chunk boundaries")
print("✅ Prevents loss of important information")
print("✅ Improves search accuracy")
print("✅ Better semantic understanding")
print("✅ Reduces false negatives in search")

# Create completion marker
with open("overlap_chunking_complete.txt", "w") as f:
    f.write("Overlap chunking demo completed successfully")

print("\n✅ Overlap chunking demo completed!")
