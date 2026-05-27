"""
Document Loading Demo - LangChain Chat with Data
Covers: PDF, URL, Notion, and YouTube loaders
"""

import os
import sys

# Fix SSL for uv-managed Python on a corporate network with Zscaler TLS inspection.
# uv bundles its own OpenSSL which doesn't trust macOS Keychain (where the Zscaler
# corporate CA lives). truststore injects macOS Keychain trust into Python's ssl module.
import truststore
truststore.inject_into_ssl()

sys.path.append('..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

# Set a user-agent to avoid warnings from WebBaseLoader
os.environ.setdefault("USER_AGENT", "langchain-demo/1.0")

# ─────────────────────────────────────────────
# 1. PDF Loading
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("1. PDF LOADER")
print("="*60)

from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("docs/India-Handbook-2024.pdf")
pages = loader.load()

print(f"Total pages loaded: {len(pages)}")

# Each page is a Document with page_content and metadata
page = pages[0]
print(f"\nFirst 500 chars of page 1:\n{page.page_content[:500]}")
print(f"\nMetadata: {page.metadata}")

# ─────────────────────────────────────────────
# 2. URL Loading
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("2. URL (WEB) LOADER")
print("="*60)

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/titles-for-programmers.md")
docs = loader.load()

print(f"Documents loaded: {len(docs)}")
print(f"\nFirst 500 chars:\n{docs[0].page_content[:500]}")
print(f"\nMetadata: {docs[0].metadata}")

# ─────────────────────────────────────────────
# 3. Notion Loading
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("3. NOTION DIRECTORY LOADER")
print("="*60)

from langchain_community.document_loaders import NotionDirectoryLoader

loader = NotionDirectoryLoader("docs/Notion_DB")
docs = loader.load()

print(f"Documents loaded: {len(docs)}")
print(f"\nFirst 200 chars:\n{docs[0].page_content[:200]}")
print(f"\nMetadata: {docs[0].metadata}")

# ─────────────────────────────────────────────
# 4. YouTube Audio Loading
#    Requires: pip install yt_dlp pydub  +  brew install ffmpeg
#    OPENAI_API_KEY must be set in .env
# ─────────────────────────────────────────────
# print("\n" + "="*60)
# print("4. YOUTUBE AUDIO LOADER")
# print("="*60)

# url = "https://www.youtube.com/watch?v=jGwO_UgTS7I"
# save_dir = "docs/youtube/"

# IDEAL WAY — works on unrestricted networks without any extra setup:
#
# from langchain_community.document_loaders import YoutubeAudioLoader
#
# loader = GenericLoader(
#     YoutubeAudioLoader([url], save_dir),
#     OpenAIWhisperParser()
# )
# docs = loader.load()
# print(f"Documents loaded: {len(docs)}")
# print(f"\nFirst 500 chars:\n{docs[0].page_content[:500]}")
#
# On corporate/restricted networks (e.g. Zscaler blocks YouTube streams),
# use the helper below which handles SSL, cookies, and fallback strategies.
# See youtube_download_helper.py for full details.

# Download + transcribe — see youtube_download_helper.py to run independently:
#
# from youtube_download_helper import download_youtube_audio, transcribe_audio
# downloaded = download_youtube_audio(url, save_dir)
# if downloaded:
#     docs = transcribe_audio(save_dir)
#     print(f"Documents loaded: {len(docs)}")
#     print(f"\nFirst 500 chars:\n{docs[0].page_content[:500]}")
