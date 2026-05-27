"""
Document Splitting Demo - LangChain Chat with Data
====================================================
Based on DeepLearning.AI "LangChain: Chat with Your Data" L2 notebook.

WHY SPLITTING MATTERS
─────────────────────
LLMs have fixed context windows.  A 300-page PDF cannot fit in one call.
Splitting turns long documents into smaller chunks that can be:
  • Stored individually in a vector DB
  • Retrieved by semantic similarity
  • Fed to an LLM without exceeding its token limit

KEY CHALLENGE — CHUNK BOUNDARY PROBLEMS
  Text split naively at a fixed size can cut across:
    - Sentences  ("The cat sat on the … mat.")
    - Code blocks (function broken mid-body)
    - Markdown sections (header separated from its content)
  Overlap and smarter splitters mitigate this.

CHUNK OVERLAP
  Overlapping N characters between adjacent chunks preserves context
  that would otherwise be lost at a boundary.

SPLITTERS COVERED
  1. CharacterTextSplitter          — split on a single separator character
  2. RecursiveCharacterTextSplitter — try multiple separators in order (default)
  3. MarkdownHeaderTextSplitter     — split on Markdown headers
  4. TokenTextSplitter              — split by tiktoken token count
  5. SentenceTransformersTokenSplitter — split by sentence-transformer tokens
  6. Language splitter              — code-aware recursive split (Python / JS / …)
  7. NLTKTextSplitter               — sentence boundaries via NLTK
  8. SpacyTextSplitter              — sentence boundaries via SpaCy

RUN
───
    cd langchain-chat-with-data
    uv run python l2_document_splitting.py

DEPENDENCIES (top-level requirements.txt + extras)
────────────────────────────────────────────────────
    langchain-text-splitters, tiktoken, sentence-transformers,
    spacy (+ en_core_web_sm model), nltk (punkt_tab corpus)
"""

from __future__ import annotations

import sys
import textwrap

import truststore
truststore.inject_into_ssl()

sys.path.append('..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# ── Shared helper ─────────────────────────────────────────────────────────────

def _show(chunks: list, label: str = "") -> None:
    """Print a summary of a chunk list — count, sizes, and first two chunks."""
    if label:
        print(f"\n  [{label}]")
    print(f"  Total chunks : {len(chunks)}")
    sizes = [len(c.page_content if hasattr(c, 'page_content') else c) for c in chunks]
    print(f"  Chunk sizes  : min={min(sizes)}  max={max(sizes)}  avg={sum(sizes)//len(sizes)}")
    for i, chunk in enumerate(chunks[:2]):
        content = chunk.page_content if hasattr(chunk, 'page_content') else chunk
        print(f"\n  ── Chunk {i+1} ──")
        print(textwrap.indent(textwrap.fill(content[:300], width=70), "  "))
        if len(content) > 300:
            print("  …[truncated]")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CharacterTextSplitter
#    Splits on a single separator (default: "\n\n").
#    chunk_size controls max characters; chunk_overlap controls the overlap.
#    Simple and fast but unaware of sentence / token boundaries.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1. CharacterTextSplitter")
print("=" * 60)

from langchain_text_splitters import CharacterTextSplitter

sample_text = """When writing documents, writers will use document structure to group content. 
This can convey to the reader, which idea's are related. For example, closely related ideas 
are in sentances. Similar ideas are under the same section header. Ideas that are related, 
but not closely, are in different sections but may reference each other.

Machine learning models like transformers can only process a fixed amount of text. 
They have a context window of a certain number of tokens. Tokens are typically 4 characters 
or 0.75 words. The context window is typically 4096 tokens for a model like GPT-3.5.

We may want to split our documents to fit within the context window. If the document is 
longer than the context window, we need to split it into multiple chunks. Each chunk can be 
processed independently or together with other chunks.
"""

# Split on double newlines (paragraph boundaries)
char_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
)
chunks = char_splitter.create_documents([sample_text])
_show(chunks)
print("\n" + "*" * 30)

# Split on single space — shows how a bad separator looks
print()
space_splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=200,
    chunk_overlap=20,
)
space_chunks = space_splitter.create_documents([sample_text])
print(f"  [separator=' '] Total chunks: {len(space_chunks)} "
      f"(less natural boundaries than '\\n\\n')")
_show(space_chunks)
print()
print("\n" + "*" * 30)

# Metadata can be passed alongside documents
metadatas = [{"source": "demo_doc"}]
chunks_with_meta = char_splitter.create_documents([sample_text], metadatas=metadatas)
print(f"\n  Chunk 1 metadata: {chunks_with_meta[0].metadata}")
_show(chunks_with_meta)
print()
print("\n" + "*" * 30)

# ══════════════════════════════════════════════════════════════════════════════
# 2. RecursiveCharacterTextSplitter
#    The RECOMMENDED general-purpose splitter.
#    Tries a list of separators in priority order:
#       ["\n\n", "\n", " ", ""]
#    Falls back to the next separator only when a chunk is still too large.
#    This keeps paragraphs → sentences → words → characters together
#    as long as possible.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. RecursiveCharacterTextSplitter  (recommended default)")
print("=" * 60)

from langchain_text_splitters import RecursiveCharacterTextSplitter

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,
    separators=["\n\n", "\n", " ", ""],   # explicit (these are the defaults)
)
r_chunks = recursive_splitter.create_documents([sample_text])
_show(r_chunks)
print("\n" + "*" * 30)

# Compare char vs recursive on a tricky string with no paragraph breaks
tricky = "abcdefghijklmnopqrstuvwxyz" * 10   # 260-char string, no whitespace

char_tricky   = CharacterTextSplitter(separator="\n\n", chunk_size=20, chunk_overlap=5)
recur_tricky  = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)

char_tricky_chunks =char_tricky.split_text(tricky)
print(f"\n  Tricky string (no whitespace, 260 chars):")
print(f"    CharacterTextSplitter   → {len(char_tricky_chunks)} chunks "
      f"(may produce 1 oversized chunk because separator not found)")
_show(char_tricky_chunks)
print("\n" + "*" * 30)

recur_tricky_chunks = recur_tricky.split_text(tricky)
print(f"    RecursiveCharTextSplitter→ {len(recur_tricky_chunks)} chunks "
      f"(always enforces chunk_size by falling back to '')")
_show(recur_tricky_chunks)
print("\n" + "*" * 30)


# ══════════════════════════════════════════════════════════════════════════════
# 3. MarkdownHeaderTextSplitter
#    Splits Markdown at specified header levels and promotes headers into
#    Document metadata so every chunk knows which section it came from.
#    Ideal for README files, wikis, course notes.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. MarkdownHeaderTextSplitter")
print("=" * 60)

from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_doc = """# Title: LangChain Overview

## Chapter 1: Introduction

LangChain is a framework for building LLM-powered applications.
It provides modular components for prompts, chains, agents, and memory.

### Section 1.1: Core Concepts

Chains connect multiple components together.
Agents use tools to decide what to do next.

## Chapter 2: Document Loading

LangChain supports many document loaders.

### Section 2.1: PDF Loader

The PyPDFLoader can load multi-page PDFs.
Each page becomes a separate Document object.

### Section 2.2: Web Loader

WebBaseLoader fetches HTML pages and strips tags.
"""

headers_to_split_on = [
    ("#",  "Header 1"),
    ("##", "Header 2"),
    ("###","Header 3"),
]

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
)
md_chunks = md_splitter.split_text(markdown_doc)

print(f"\n  Total chunks: {len(md_chunks)}")
for i, chunk in enumerate(md_chunks):
    print(f"\n  ── Chunk {i+1} ──")
    print(f"  Metadata : {chunk.metadata}")
    print(f"  Content  : {chunk.page_content[:120]!r}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. TokenTextSplitter
#    Splits on token counts using tiktoken (OpenAI's tokenizer).
#    Important when your downstream LLM charges by token or has a strict
#    token-based context window.  chunk_size is in *tokens*, not characters.
#    Rule of thumb: 1 token ≈ 4 characters or 0.75 words.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. TokenTextSplitter  (tiktoken)")
print("=" * 60)

from langchain_text_splitters import TokenTextSplitter

token_splitter = TokenTextSplitter(
    encoding_name="cl100k_base",  # GPT-3.5 / GPT-4 encoding
    chunk_size=30,                # tokens per chunk
    chunk_overlap=5,
)
token_chunks = token_splitter.create_documents([sample_text])
_show(token_chunks)

print(f"\n  NOTE: chunk sizes above are in *characters*.  "
      f"Each chunk here is ≤30 tokens ≈ 120 chars.")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SentenceTransformersTokenTextSplitter
#    Like TokenTextSplitter but uses the sentence-transformers tokenizer
#    (Hugging Face).  Useful when you are embedding with a sentence-transformer
#    model and want chunks that fit its token limit exactly.
#    Default model: sentence-transformers/all-mpnet-base-v2 (max 384 tokens).
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. SentenceTransformersTokenTextSplitter")
print("=" * 60)

from langchain_text_splitters import SentenceTransformersTokenTextSplitter

st_splitter = SentenceTransformersTokenTextSplitter(
    model_name="sentence-transformers/all-MiniLM-L6-v2",   # 256-token limit
    chunk_size=50,      # tokens
    chunk_overlap=10,
)
st_chunks = st_splitter.create_documents([sample_text])
_show(st_chunks)

# Count tokens on a single string
token_count = st_splitter.count_tokens(text=sample_text)
print(f"\n  Token count for full sample_text : {token_count} "
      f"(sentence-transformer tokenizer)")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Language splitter  (RecursiveCharacterTextSplitter.from_language)
#    Code-aware variant of RecursiveCharacterTextSplitter.
#    Uses language-specific separators (class/def/function keywords, braces …)
#    so splits happen at natural code boundaries rather than mid-expression.
#    Supported: PYTHON, JS, TS, RUBY, RUST, GO, CPP, JAVA, SCALA, SWIFT, MARKDOWN …
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. Language Splitter  (Python & JavaScript)")
print("=" * 60)

from langchain_text_splitters import Language

python_code = '''
def hello_world():
    """Print a greeting."""
    print("Hello, World!")

class Greeter:
    """A simple greeter class."""

    def __init__(self, name: str):
        self.name = name

    def greet(self) -> str:
        return f"Hello, {self.name}!"

    def farewell(self) -> str:
        return f"Goodbye, {self.name}!"

def main():
    greeter = Greeter("LangChain")
    print(greeter.greet())
    print(greeter.farewell())

if __name__ == "__main__":
    main()
'''

js_code = '''
function helloWorld() {
    console.log("Hello, World!");
}

class Greeter {
    constructor(name) {
        this.name = name;
    }

    greet() {
        return `Hello, ${this.name}!`;
    }

    farewell() {
        return `Goodbye, ${this.name}!`;
    }
}

function main() {
    const g = new Greeter("LangChain");
    console.log(g.greet());
}

main();
'''

# Show what separators the Language splitter uses for Python
py_separators = RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)
print(f"\n  Python separators : {py_separators[:5]} …")

py_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=20,
)
py_chunks = py_splitter.create_documents([python_code])
_show(py_chunks, "Python")

js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS,
    chunk_size=200,
    chunk_overlap=20,
)
js_chunks = js_splitter.create_documents([js_code])
_show(js_chunks, "JavaScript")


# ══════════════════════════════════════════════════════════════════════════════
# 7. NLTKTextSplitter
#    Uses NLTK's Punkt sentence tokenizer to find sentence boundaries, then
#    accumulates sentences into chunks that respect chunk_size.
#    Produces more natural chunk boundaries than character-based splitting.
#    Requires: pip install nltk  +  nltk.download('punkt_tab')
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. NLTKTextSplitter")
print("=" * 60)

import nltk
nltk.download("punkt_tab", quiet=True)

from langchain_text_splitters import NLTKTextSplitter

nltk_text = (
    "Dr. Smith went to Washington D.C. last week. "
    "He met with Mr. Jones from the U.S. Dept. of Energy. "
    "They discussed renewable energy. "
    "Solar panels are becoming cheaper every year. "
    "Wind turbines are also improving rapidly. "
    "Electric vehicles are on the rise. "
    "Battery technology is a key bottleneck. "
    "Scientists are working on solid-state batteries. "
    "These could double energy density. "
    "The future looks bright for clean energy."
)

nltk_splitter = NLTKTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
)
nltk_chunks = nltk_splitter.create_documents([nltk_text])
_show(nltk_chunks)

print(f"\n  NLTK correctly keeps 'Dr. Smith', 'D.C.', 'U.S.' as non-sentence-ends.")


# ══════════════════════════════════════════════════════════════════════════════
# 8. SpacyTextSplitter
#    Like NLTKTextSplitter but uses spaCy's sentence boundary detection.
#    spaCy is typically more accurate on complex or domain-specific text.
#    Requires: pip install spacy  +  python -m spacy download en_core_web_sm
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("8. SpacyTextSplitter")
print("=" * 60)

from langchain_text_splitters import SpacyTextSplitter

spacy_splitter = SpacyTextSplitter(
    pipeline="en_core_web_sm",
    chunk_size=200,
    chunk_overlap=20,
)
spacy_chunks = spacy_splitter.create_documents([nltk_text])
_show(spacy_chunks)

print(f"\n  spaCy also handles abbreviations correctly via its NLP pipeline.")


# ══════════════════════════════════════════════════════════════════════════════
# 9. Real-world pipeline: PDF → RecursiveCharacterTextSplitter
#    Load a real PDF with PyPDFLoader, then split all pages into overlapping
#    chunks ready for embedding / vector storage.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("9. Real-world: PDF → RecursiveCharacterTextSplitter")
print("=" * 60)

import os
from langchain_community.document_loaders import PyMuPDFLoader

PDF_PATH = "docs/India-Handbook-2024.pdf"

if os.path.exists(PDF_PATH):
    loader = PyMuPDFLoader(PDF_PATH)
    pages  = loader.load()
    print(f"\n  PDF pages loaded: {len(pages)}")

    pdf_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    pdf_chunks = pdf_splitter.split_documents(pages)
    print(f"  Chunks after splitting: {len(pdf_chunks)}")
    print(f"  First chunk metadata  : {pdf_chunks[0].metadata}")
    print(f"  First chunk (200 chars): {pdf_chunks[0].page_content[:200]!r}")
else:
    print(f"\n  [SKIP] {PDF_PATH} not found — place a PDF there to run this section.")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY — Choosing a splitter")
print("=" * 60)
summary = """
Use case                         → Recommended splitter
──────────────────────────────────────────────────────────────
General text / PDFs              → RecursiveCharacterTextSplitter
Token-limited LLMs (OpenAI)      → TokenTextSplitter (tiktoken)
Sentence-transformer embeddings  → SentenceTransformersTokenTextSplitter
Markdown / READMEs / wikis       → MarkdownHeaderTextSplitter
Source code (Python / JS / …)    → Language splitter
Natural sentence boundaries      → NLTKTextSplitter or SpacyTextSplitter
"""
print(summary)
