"""
gem_utils.py

Shared utilities for the GEM RAG pipeline.
Imported by all scripts to eliminate duplication.

Covers:
- Language detection (text-based + URL-based)
- Arabic text detection
- URL categorization
- Token counting
- Context truncation
- Embedding prefix helpers
"""

import re
import unicodedata
from typing import Optional
from urllib.parse import urlparse


# ═════════════════════════════════════════════
# Language Detection
# ═════════════════════════════════════════════

def detect_language_from_url(url: str) -> str:
    """
    Detect language from URL prefix.
    Most reliable method for gem.eg (explicit /ar/ or /en/ prefix).
    """
    path = urlparse(url).path
    return "ar" if (path.startswith("/ar") or path.startswith("/ar/")) else "en"


def detect_language_from_text(text: str, threshold: float = 0.2) -> str:
    """
    Detect language from text content.
    Uses Arabic Unicode block coverage as signal.
    Returns 'ar' or 'en'.
    """
    if not text:
        return "en"
    return "ar" if is_arabic_text(text, threshold) else "en"


# ═════════════════════════════════════════════
# Arabic Text Helpers
# ═════════════════════════════════════════════

def is_arabic_text(text: str, threshold: float = 0.2) -> bool:
    """
    Return True if more than `threshold` fraction of characters
    fall in any Arabic Unicode block.
    """
    if not text:
        return False
    arabic_count = sum(
        1 for c in text
        if "\u0600" <= c <= "\u06ff"   # Arabic
        or "\u0750" <= c <= "\u077f"   # Arabic Supplement
        or "\ufb50" <= c <= "\ufdff"   # Arabic Presentation Forms-A
        or "\ufe70" <= c <= "\ufeff"   # Arabic Presentation Forms-B
    )
    return arabic_count / max(len(text), 1) > threshold


# ═════════════════════════════════════════════
# URL Utilities
# ═════════════════════════════════════════════

_CATEGORY_MAP: dict[str, list[str]] = {
    "collection":   ["collection", "artefact", "artifact"],
    "visitor_info": ["visit", "ticket", "hour", "plan", "map", "tip", "access"],
    "educational":  ["education", "children", "arts-and-crafts"],
    "events":       ["event", "whats-on", "programme", "hackathon"],
    "about":        ["about", "research", "conservation"],
    "exhibition":   ["exhibition", "gallery", "galleries", "hall", "stairs"],
}


def categorize_url(url: str) -> str:
    """Map a URL to its content category."""
    url_lower = url.lower()
    for category, keywords in _CATEGORY_MAP.items():
        if any(kw in url_lower for kw in keywords):
            return category
    return "general"


# ═════════════════════════════════════════════
# Token Counting & Context Truncation
# ═════════════════════════════════════════════

def get_tokenizer(model_name: str = "gpt-4o-mini"):
    """
    Return a tiktoken tokenizer.
    Falls back to cl100k_base if model name is unrecognised.
    """
    try:
        import tiktoken
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, tokenizer=None) -> int:
    """Count tokens in text using tiktoken."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))


def truncate_context(
    context: str,
    max_tokens: int = 3_000,
    tokenizer=None,
) -> str:
    """
    Truncate context string to max_tokens.
    Adds a trailing note when truncation occurs.
    Protects the LLM call from context-window overflow.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    tokens = tokenizer.encode(context)
    if len(tokens) <= max_tokens:
        return context
    truncated = tokenizer.decode(tokens[:max_tokens])
    return truncated + "\n\n[... context truncated to fit model window ...]"


# ═════════════════════════════════════════════
# Embedding Prefix Helpers (multilingual-e5)
# ═════════════════════════════════════════════

def e5_passage_prefix(text: str) -> str:
    """
    Prepend the required 'passage: ' prefix for multilingual-e5 models.
    Applied to ALL document chunks at embedding time.
    """
    return f"passage: {text}"


def e5_query_prefix(query: str) -> str:
    """
    Prepend the required 'query: ' prefix for multilingual-e5 models.
    Applied to user queries at retrieval time.
    """
    return f"query: {query}"