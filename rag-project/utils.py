#!/usr/bin/env python3
"""
Common utilities for RAG search methods
"""

import os
import glob

def read_techcorp_docs():
    """Read all documents from techcorp-docs directory"""
    docs = []
    doc_paths = []
    
    # Try different possible paths
    possible_paths = [
        "/home/lab-user/techcorp-docs/**/*.md",  # Lab environment
        "techcorp-docs/**/*.md",                  # Local development
        "./techcorp-docs/**/*.md"                # Current directory
    ]
    
    files = []
    for pattern in possible_paths:
        files = glob.glob(pattern, recursive=True)
        if files:
            break
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # Only add non-empty files
                    docs.append(content)
                    doc_paths.append(file_path)
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
    
    return docs, doc_paths

def get_doc_info():
    """Get document information for display"""
    docs, paths = read_techcorp_docs()
    
    print(f"📚 Loaded {len(docs)} documents")
    print("\nDocuments:")
    for i, (doc, path) in enumerate(zip(docs, paths)):
        # Get relative path for cleaner display
        rel_path = path.replace("/home/lab-user/techcorp-docs/", "")
        print(f"{i+1}. [{rel_path}] {doc[:80]}{'...' if len(doc) > 80 else ''}")
    
    return docs, paths
