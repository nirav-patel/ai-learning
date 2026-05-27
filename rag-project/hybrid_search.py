#!/usr/bin/env python3
"""
Hybrid Search Implementation
Combines TF-IDF and BM25 scores with different weights
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import re
from utils import get_doc_info

def hybrid_search(query, docs, tfidf_weight=0.3, bm25_weight=0.7):
    """Combine TF-IDF and BM25 scores with weights"""
    
    # TF-IDF scores
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    query_vector = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # BM25 scores
    tokenized_docs = [re.sub(r'[^a-zA-Z\s]', '', doc.lower()).split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = re.sub(r'[^a-zA-Z\s]', '', query.lower()).split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Normalize BM25 scores to 0-1 range for fair comparison
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()
    
    # Combine scores
    hybrid_scores = tfidf_weight * tfidf_scores + bm25_weight * bm25_scores
    
    return tfidf_scores, bm25_scores, hybrid_scores

def main():
    """Main function to demonstrate hybrid search"""
    print("🔍 Hybrid Search Demo")
    print("=" * 50)
    
    # Load documents from techcorp-docs
    docs, doc_paths = get_doc_info()
    
    # Test different weight combinations
    query = "remote work policy"
    print(f"🔎 Testing query: '{query}'")
    print("=" * 50)
    
    weight_combinations = [
        (0.5, 0.5, "Equal weights"),
        (0.3, 0.7, "BM25 favored"),
        (0.7, 0.3, "TF-IDF favored")
    ]
    
    for tfidf_w, bm25_w, description in weight_combinations:
        print(f"\n📊 {description} (TF-IDF: {tfidf_w}, BM25: {bm25_w})")
        print("-" * 40)
        
        tfidf_scores, bm25_scores, hybrid_scores = hybrid_search(query, docs, tfidf_w, bm25_w)
        
        # Get top 3 results
        top_indices = hybrid_scores.argsort()[-3:][::-1]
        
        print("Top 3 results:")
        for i, idx in enumerate(top_indices, 1):
            # Show only document path and score
            doc_name = doc_paths[idx].split('/')[-1]  # Just the filename
            print(f"  {i}. Score: {hybrid_scores[idx]:.4f} - {doc_name}")
    
    print(f"\n✅ Hybrid search analysis completed!")

if __name__ == "__main__":
    main()
