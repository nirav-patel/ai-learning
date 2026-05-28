"""ingestion.py — DocumentLoader: PDF loading and token-based chunking.

Uses pymupdf4llm instead of raw PyMuPDF to extract Markdown-rich content.
pymupdf4llm preserves:
  - Table structure (rendered as Markdown pipe tables)
  - Headings and bold text (as Markdown ##, **)
  - Multi-column reading order
  - Image captions (images noted as placeholders, not silently dropped)

Splitting uses RecursiveCharacterTextSplitter with Markdown-aware separators
so chunk boundaries prefer header/paragraph boundaries over mid-sentence cuts.
"""
from __future__ import annotations

import logging
import os

import pymupdf4llm
from langchain_core.documents import Document

from .config import AppConfig

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load PDFs as Markdown-rich text and split into token-sized chunks.

    Args:
        config: AppConfig supplying chunk_size, chunk_overlap, tiktoken_encoding,
                and data_sources_dir.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def load_pdf(self, path: str) -> list[Document]:
        """Load a single PDF via pymupdf4llm and return token-sized Document chunks.

        Each page is extracted as Markdown (tables → pipe tables, headings → ##).
        The splitter uses Markdown-aware separators so splits prefer heading and
        paragraph boundaries over mid-sentence cuts.

        Args:
            path: Absolute or relative path to the PDF file.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # page_chunks=True → one dict per page with 'text' + 'metadata'
        raw_pages = pymupdf4llm.to_markdown(path, page_chunks=True)

        pages: list[Document] = []
        for raw in raw_pages:
            meta = raw.get("metadata", {})
            doc = Document(
                page_content=raw.get("text", ""),
                metadata={
                    "source": path,
                    "page":   meta.get("page_number", 0),
                    "title":  meta.get("title")  or "",
                    "author": meta.get("author") or "",
                },
            )
            self._sanitize_metadata(doc)
            pages.append(doc)

        # Markdown-aware separators: split at headers first, then paragraphs,
        # then sentences, falling back to characters only as last resort
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=self._config.tiktoken_encoding,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
            separators=[
                "\n## ", "\n### ", "\n#### ",   # Markdown headings
                "\n\n",                          # paragraph breaks
                "\n",                            # line breaks
                ". ", "! ", "? ",               # sentence ends
                " ", "",                         # last resort
            ],
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
