"""
gem_chunker.py

Splits cleaned documents into token-sized chunks for RAG.
— Modal compatible
— Arabic + English bilingual
— Volume: "rag"

Fixes over v1:
- is_heading_line improved (no more istitle/isupper false negatives)
- Title prefix token cost subtracted from effective chunk size
- chunk_id uses URL hash (safe as DB key)
- Shared utility functions inlined
- Chunk overlap raised to 80 tokens (denser museum content)
"""

import modal
from pathlib import Path

app = modal.App("gem-chunker")

chunker_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "langchain-text-splitters",
        "tiktoken",
    ])
)

volume      = modal.Volume.from_name("rag", create_if_missing=True)
VOLUME_PATH = "/data"


@app.function(
    image=chunker_image,
    volumes={VOLUME_PATH: volume},
    timeout=1800,
    memory=4096,
)
def run_chunker(
    chunk_size:     int = 400,
    chunk_overlap:  int = 80,    # raised: museum artifact text is dense
    min_chunk_size: int = 30,
):
    import json
    import hashlib
    import logging
    from typing import List, Dict
    from pathlib import Path
    from collections import Counter

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import tiktoken

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # ─────────────────────────────────────────
    # Shared helpers (inlined for Modal)
    # ─────────────────────────────────────────
    def is_arabic_text(text: str, threshold: float = 0.2) -> bool:
        if not text:
            return False
        count = sum(
            1 for c in text
            if "\u0600" <= c <= "\u06ff"
            or "\u0750" <= c <= "\u077f"
            or "\ufb50" <= c <= "\ufdff"
            or "\ufe70" <= c <= "\ufeff"
        )
        return count / max(len(text), 1) > threshold

    def make_chunk_id(url: str, index: int) -> str:
        """
        URL-safe chunk identifier using MD5 hash of URL.
        Avoids special characters in IDs used as DB keys.
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
        return f"{url_hash}__chunk_{index}"

    # ─────────────────────────────────────────
    # DocumentChunker
    # ─────────────────────────────────────────
    class DocumentChunker:

        def __init__(
            self,
            chunk_size:     int = 400,
            chunk_overlap:  int = 80,
            min_chunk_size: int = 30,
            model_name:     str = "gpt-4o-mini",
        ):
            self.chunk_size     = chunk_size
            self.chunk_overlap  = chunk_overlap
            self.min_chunk_size = min_chunk_size

            try:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            except Exception:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

            # English splitter
            self.en_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
                length_function=self.count_tokens,
            )

            # Arabic splitter — includes Arabic punctuation separators
            self.ar_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[
                    "\n\n", "\n",
                    "۔ ", "। ",    # Arabic/Urdu full stops
                    "، ",           # Arabic comma
                    ". ", "! ", "? ", "؟ ",
                    " ", "",
                ],
                length_function=self.count_tokens,
            )

        def count_tokens(self, text: str) -> int:
            return len(self.tokenizer.encode(text))

        def is_heading_line(self, line: str, language: str = "en") -> bool:
            """
            Detect section headings.

            v1 used istitle()/isupper() which missed most real headings.
            v2 uses word count + capitalisation + punctuation heuristics.
            """
            line = line.strip()
            if not line:
                return False

            if language == "ar":
                ar_end_punct = {".", "،", "۔", "!", "?", "؟"}
                return (
                    len(line) < 80
                    and not any(line.endswith(p) for p in ar_end_punct)
                    and is_arabic_text(line)
                )
            else:
                words = line.split()
                return (
                    5 < len(line) < 120     # reasonable heading length
                    and len(words) <= 14    # headings are concise
                    and not line.endswith(".")
                    and line[0].isupper()   # starts with capital letter
                )

        def split_by_headings(
            self, content: str, language: str = "en"
        ) -> List[str]:
            """Split content on detected heading lines."""
            lines    = content.split("\n")
            sections = []
            current  = []

            for line in lines:
                if self.is_heading_line(line, language) and current:
                    sections.append("\n".join(current))
                    current = [line]
                else:
                    current.append(line)

            if current:
                sections.append("\n".join(current))

            return [s.strip() for s in sections if s.strip()]

        def chunk_text(self, text: str, language: str = "en") -> List[str]:
            if self.count_tokens(text) <= self.chunk_size:
                return [text]
            splitter = self.ar_splitter if language == "ar" else self.en_splitter
            return splitter.split_text(text)

        def build_title_prefix(self, title: str, language: str) -> str:
            if language == "ar":
                return f"العنوان: {title}\n\n"
            return f"Title: {title}\n\n"

        def chunk_document(self, document: Dict) -> List[Dict]:
            content  = document.get("content", "")
            title    = document.get("title", "")
            language = document.get("language", "en")
            url      = document.get("url", "")

            if not content:
                return []

            # Pre-compute prefix cost so we don't overflow chunk_size
            prefix       = self.build_title_prefix(title, language)
            prefix_tokens = self.count_tokens(prefix)

            # Temporarily shrink splitter chunk_size to account for prefix
            original_size = self.chunk_size
            effective_size = max(original_size - prefix_tokens, 50)

            if language == "ar":
                self.ar_splitter._chunk_size = effective_size
            else:
                self.en_splitter._chunk_size = effective_size

            # 1. Split by headings
            sections    = self.split_by_headings(content, language)
            chunks_text = []

            for section in sections:
                if not section.strip():
                    continue
                if self.count_tokens(section) <= effective_size:
                    chunks_text.append(section)
                else:
                    chunks_text.extend(self.chunk_text(section, language))

            # Restore splitter size
            self.ar_splitter._chunk_size = original_size
            self.en_splitter._chunk_size = original_size

            # 2. Filter below min token threshold
            chunks_text = [
                c for c in chunks_text
                if self.count_tokens(c) >= self.min_chunk_size
            ]

            if not chunks_text:
                return []

            # 3. Prepend title prefix to each chunk
            titled = []
            for chunk in chunks_text:
                full = prefix + chunk if not chunk.startswith(prefix) else chunk
                titled.append(full)

            # 4. Build metadata
            total  = len(titled)
            chunks = []
            for idx, text in enumerate(titled):
                chunks.append({
                    "chunk_id":       make_chunk_id(url, idx),
                    "title":          title,
                    "url":            url,
                    "category":       document.get("category", "general"),
                    "date":           document.get("date", ""),
                    "language":       language,
                    "content":        text,
                    "chunk_index":    idx,
                    "total_chunks":   total,
                    "is_first_chunk": idx == 0,
                    "is_last_chunk":  idx == total - 1,
                    "token_count":    self.count_tokens(text),
                    "char_count":     len(text),
                })

            return chunks

        def chunk_all(
            self,
            documents: List[Dict],
            output_path: str,
        ) -> List[Dict]:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            all_chunks = []

            for i, doc in enumerate(documents):
                doc_chunks = self.chunk_document(doc)
                all_chunks.extend(doc_chunks)
                lang = doc.get("language", "en").upper()
                logger.info(
                    f"[{lang}] {i+1}/{len(documents)} "
                    f"'{doc.get('title','')[:40]}' "
                    f"→ {len(doc_chunks)} chunks"
                )

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, indent=2, ensure_ascii=False)

            avg_tokens = (
                sum(c["token_count"] for c in all_chunks)
                / max(len(all_chunks), 1)
            )
            logger.info(
                f"Total: {len(all_chunks)} chunks | "
                f"Avg tokens: {avg_tokens:.1f}"
            )
            return all_chunks

    # ─────────────────────────────────────────
    # Load
    # ─────────────────────────────────────────
    input_path = Path(VOLUME_PATH) / "cleaned_documents.json"
    if not input_path.exists():
        raise FileNotFoundError(
            "cleaned_documents.json not found. Run the cleaner first."
        )

    with open(input_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    lang_counts = Counter(d.get("language", "en") for d in documents)
    print(f"📄 Loaded {len(documents)} cleaned documents")
    print(f"🌐 English : {lang_counts.get('en', 0)}")
    print(f"🌐 Arabic  : {lang_counts.get('ar', 0)}")

    # ─────────────────────────────────────────
    # Chunk
    # ─────────────────────────────────────────
    chunker     = DocumentChunker(chunk_size, chunk_overlap, min_chunk_size)
    output_path = str(Path(VOLUME_PATH) / "all_chunks.json")
    chunks      = chunker.chunk_all(documents, output_path)

    volume.commit()

    lang_chunk_counts = Counter(c.get("language", "en") for c in chunks)
    avg_tokens        = (
        sum(c["token_count"] for c in chunks) / max(len(chunks), 1)
    )

    print(f"\n✅ Chunking Complete!")
    print(f"📦 Total chunks   : {len(chunks)}")
    print(f"   English        : {lang_chunk_counts.get('en', 0)}")
    print(f"   Arabic         : {lang_chunk_counts.get('ar', 0)}")
    print(f"   Avg tokens     : {avg_tokens:.1f}")
    print(f"💾 Saved to       : {output_path}")

    for lang in ("en", "ar"):
        sample = next((c for c in chunks if c.get("language") == lang), None)
        if sample:
            flag = "🇬🇧" if lang == "en" else "🇪🇬"
            print(f"\n{flag} {lang.upper()} Sample:")
            print(f"   ID      : {sample['chunk_id']}")
            print(f"   Tokens  : {sample['token_count']}")
            print(f"   Preview : {sample['content'][:200]}...")

    return {
        "total_chunks":   len(chunks),
        "english_chunks": lang_chunk_counts.get("en", 0),
        "arabic_chunks":  lang_chunk_counts.get("ar", 0),
        "avg_tokens":     round(avg_tokens, 1),
        "saved_to":       output_path,
    }


@app.local_entrypoint()
def main():
    print("🚀 Launching GEM Chunker on Modal...")
    result = run_chunker.remote(
        chunk_size=400, chunk_overlap=80, min_chunk_size=30
    )
    print(f"\n🎉 Done!")
    print(f"   Total   : {result['total_chunks']}")
    print(f"   English : {result['english_chunks']}")
    print(f"   Arabic  : {result['arabic_chunks']}")
    print(f"   Avg tok : {result['avg_tokens']}")
    print(f"   Saved   : {result['saved_to']}")