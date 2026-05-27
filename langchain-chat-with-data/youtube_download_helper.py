"""
YouTube Audio Download Helper
==============================
This module handles YouTube audio download with multiple fallback strategies.
It exists because the ideal one-liner (YoutubeAudioLoader) doesn't work in
all environments due to:

  1. Corporate Zscaler TLS inspection — uv/pyenv Python doesn't trust the
     corporate CA by default. Fixed with `truststore`.
  2. YouTube bot-detection (HTTP 403) — requires browser cookies exported
     from Chrome via "Get cookies.txt LOCALLY" extension.
  3. YouTube n-challenge (yt-dlp 2026+) — requires a JS runtime (deno/node)
     to solve, which may be blocked on corporate networks anyway.
  4. YoutubeAudioLoader has no API to pass yt-dlp options (cookiefile, etc.)
     so we call yt_dlp.YoutubeDL directly.

IDEAL WAY (works on unrestricted networks, no cookies needed):
--------------------------------------------------------------
    from langchain_community.document_loaders import YoutubeAudioLoader
    from langchain_community.document_loaders.generic import GenericLoader
    from langchain_community.document_loaders.parsers import OpenAIWhisperParser

    loader = GenericLoader(
        YoutubeAudioLoader([url], save_dir),
        OpenAIWhisperParser()
    )
    docs = loader.load()

HOW TO CREATE cookies.txt (needed on corporate/restricted networks):
--------------------------------------------------------------------
  1. Install Chrome extension "Get cookies.txt LOCALLY"
     https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc
  2. Open https://www.youtube.com in Chrome (sign in recommended)
  3. Click the extension icon → Export → save as docs/youtube/cookies.txt
  4. Call download_youtube_audio() — Strategy 1 will pick it up automatically
"""

import glob
import os

import yt_dlp
from langchain_community.document_loaders.generic import GenericLoader, FileSystemBlobLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser


def download_youtube_audio(url: str, save_dir: str, cookies_file: str = None) -> bool:
    """
    Download YouTube audio to save_dir using three fallback strategies.

    Returns True if an audio file is ready in save_dir (downloaded now or
    already present from a previous run), False otherwise.
    """
    os.makedirs(save_dir, exist_ok=True)

    if cookies_file is None:
        cookies_file = os.path.join(save_dir, "cookies.txt")

    def _opts(extra: dict = None) -> dict:
        base = {
            "format": "m4a/bestaudio/best",
            "noplaylist": True,
            "outtmpl": os.path.join(save_dir, "%(title)s.%(ext)s"),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}],
            "quiet": True,
            "no_warnings": False,
        }
        if extra:
            base.update(extra)
        return base

    print(f"Downloading audio from: {url}")

    # Strategy 1: cookies.txt file (most portable; works on macOS without sandbox issues)
    if os.path.exists(cookies_file):
        try:
            with yt_dlp.YoutubeDL(_opts({"cookiefile": cookies_file})) as ydl:
                ydl.download([url])
            print("Downloaded using cookies.txt.")
            return True
        except Exception as e:
            print(f"  cookies.txt failed ({type(e).__name__}), trying Chrome cookies...")

    # Strategy 2: live Chrome browser cookies (may fail if Chrome is sandboxed or if
    # corporate proxy blocks the stream even with valid cookies)
    try:
        with yt_dlp.YoutubeDL(_opts({"cookiesfrombrowser": ("chrome",)})) as ydl:
            ydl.download([url])
        print("Downloaded using Chrome browser cookies.")
        return True
    except Exception as e:
        print(f"  Chrome cookies failed ({type(e).__name__}), trying TLS impersonation...")

    # Strategy 3: curl_cffi TLS impersonation — mimics a real browser TLS handshake
    # without needing cookies. Requires: pip install curl_cffi
    try:
        with yt_dlp.YoutubeDL(_opts({"impersonate": "chrome"})) as ydl:
            ydl.download([url])
        print("Downloaded using browser TLS impersonation.")
        return True
    except Exception as e:
        print(f"  All auto-download strategies failed: {type(e).__name__}")
        print()
        print("  Likely cause: corporate network (Zscaler) is blocking YouTube media streams.")
        print("  To run the transcription demo anyway, manually place an .m4a or .mp3 file in:")
        print(f"    {os.path.abspath(save_dir)}")
        print("  Then re-run this script.")

    # Fallback: use a pre-existing audio file already in save_dir
    existing = glob.glob(os.path.join(save_dir, "*.m4a")) + \
               glob.glob(os.path.join(save_dir, "*.mp3"))
    if existing:
        print(f"\nUsing existing audio file for transcription demo: {existing[0]}")
        return True

    return False


def transcribe_audio(save_dir: str) -> list:
    """Transcribe all .m4a files in save_dir using OpenAI Whisper."""
    loader = GenericLoader(
        FileSystemBlobLoader(save_dir, glob="*.m4a"),
        OpenAIWhisperParser()
    )
    return loader.load()


if __name__ == "__main__":
    import truststore
    truststore.inject_into_ssl()

    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    _url = "https://www.youtube.com/watch?v=jGwO_UgTS7I"
    _save_dir = "docs/youtube/"

    if download_youtube_audio(_url, _save_dir):
        docs = transcribe_audio(_save_dir)
        print(f"\nDocuments loaded: {len(docs)}")
        print(f"\nFirst 500 chars:\n{docs[0].page_content[:500]}")
