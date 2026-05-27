"""PDF loading and token-based splitting.

PyMuPDFLoader is used in preference to PyPDFLoader because it:
  - Preserves table cell order and multi-column reading sequences
  - Extracts richer per-page metadata (author, title, page dimensions)
  - Handles complex layouts (rotated text, sidebars) more reliably

Splitting uses RecursiveCharacterTextSplitter.from_tiktoken_encoder so that
chunk boundaries always fall on BPE token edges — never mid-token — which
avoids embedding quality degradation on sub-word fragments.
"""
from __future__ import annotations

import os

from .config import AppConfig


def _sanitize_doc_metadata(doc) -> None:
    """Normalize metadata fields that break Weaviate schema ingestion."""
    for key in ("creationDate", "creationdate", "modDate", "moddate"):
        doc.metadata.pop(key, None)


def split_pdf(path: str, config: AppConfig) -> list:
    """Load a single PDF and split it into token-sized chunks.

    Args:
        path:   Absolute or relative path to the PDF file.
        config: AppConfig containing chunk_size, chunk_overlap, tiktoken_encoding.

    Returns:
        List of LangChain Document objects, one per chunk.
    """
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    pages    = PyMuPDFLoader(path).load()
    for page in pages:
        _sanitize_doc_metadata(page)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=config.tiktoken_encoding,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    return splitter.split_documents(pages)


def split_pdfs_from_dir(directory: str, config: AppConfig) -> tuple[list, list[str]]:
    """Load and split all *.pdf files found directly inside *directory*.

    Args:
        directory: Path to a directory that contains PDF files.
        config:    AppConfig for splitting parameters.

    Returns:
        (all_splits, pdf_paths) — combined chunk list and the paths that were loaded.
    """
    if not os.path.isdir(directory):
        return [], []

    pdf_paths = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".pdf")
    ]

    all_splits: list = []
    for path in pdf_paths:
        fname = os.path.basename(path)
        print(f"  → {fname}")
        all_splits.extend(split_pdf(path, config))

    return all_splits, pdf_paths
