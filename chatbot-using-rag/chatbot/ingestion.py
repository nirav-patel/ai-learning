"""ingestion.py — DocumentLoader: PDF loading and token-based chunking.

PyMuPDF is preferred over PyPDFLoader because it:
  - Preserves table cell order and multi-column reading sequences
  - Extracts richer per-page metadata (author, title, page dimensions)
  - Handles complex layouts (rotated text, sidebars) more reliably

Splitting uses RecursiveCharacterTextSplitter.from_tiktoken_encoder so that
chunk boundaries always fall on BPE token edges — never mid-token.
"""
from __future__ import annotations

import logging
import os

import pymupdf
from langchain_core.documents import Document

from .config import AppConfig

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load PDFs and split them into token-sized chunks.

    Args:
        config: AppConfig supplying chunk_size, chunk_overlap, tiktoken_encoding,
                and data_sources_dir.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def load_pdf(self, path: str) -> list[Document]:
        """Load a single PDF and return a list of token-sized Document chunks.

        Args:
            path: Absolute or relative path to the PDF file.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        pages: list[Document] = []
        with pymupdf.open(path) as pdf:
            meta = pdf.metadata or {}
            for idx, page in enumerate(pdf, start=1):
                pages.append(
                    Document(
                        page_content=page.get_text("text"),
                        metadata={
                            "source": path,
                            "page": idx,
                            "title":  meta.get("title")  or "",
                            "author": meta.get("author") or "",
                        },
                    )
                )

        for page in pages:
            self._sanitize_metadata(page)

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=self._config.tiktoken_encoding,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )
        return splitter.split_documents(pages)

    def load_directory(self) -> tuple[list[Document], list[str]]:
        """Load and chunk all *.pdf files in config.data_sources_dir.

        Returns:
            (all_chunks, pdf_paths) — combined chunk list and source paths.
        """
        directory = self._config.data_sources_dir
        if not os.path.isdir(directory):
            return [], []

        pdf_paths = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(".pdf")
        ]

        all_chunks: list[Document] = []
        for path in pdf_paths:
            logger.debug("Loading: %s", os.path.basename(path))
            all_chunks.extend(self.load_pdf(path))

        return all_chunks, pdf_paths

    @staticmethod
    def _sanitize_metadata(doc: Document) -> None:
        """Remove date fields that break Weaviate schema ingestion."""
        for key in ("creationDate", "creationdate", "modDate", "moddate"):
            doc.metadata.pop(key, None)
