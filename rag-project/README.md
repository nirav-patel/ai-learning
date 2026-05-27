uv pip install -r requirements.txt 
uv pip install "sentence-transformers==2.2.2" "huggingface_hub<0.20" openai
uv pip install sentence-transformers
uv pip install chromadb  
uv pip install chromadb langchain langchain-openai langchain-text-splitters langchain-core spacy sentence-transformers

uv run <file_name.py>

uv run bm25_search.py
uv run tfidf_search.py 
uv run semantic_search_demo.py

# Chunking Tests
uv pip install chromadb langchain langchain-openai langchain-text-splitters langchain-core spacy sentence-transformers
python verify_environment.py
uv run python chunking_problem_demo.py
uv run python basic_chunking.py
uv run python overlap_chunking.py
uv run python sentence_chunking.py
uv run python chunked_search.py

source ../.bash_profile  
uv run python agentic_chunking_demo.py