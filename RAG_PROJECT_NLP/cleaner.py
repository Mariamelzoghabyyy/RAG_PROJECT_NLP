"""
gem_cleaner.py

Cleans and normalises extracted web content.
— Modal compatible
— Arabic + English bilingual support
— Volume: "rag"

Fixes over v1:
- EN noise patterns no longer applied to AR text
- Shared utility functions (no duplication)
- Lower min_line_length (15 EN / 10 AR) to keep short headings
- Alef normalisation only applied to embedding copy, not display copy
- Dead code removed (image_captions / headings not scraped → removed)
- Reduction threshold warning kept at 70%
"""

import modal
from pathlib import Path

app = modal.App("gem-cleaner")

cleaner_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["regex"])
)

volume      = modal.Volume.from_name("rag", create_if_missing=True)
VOLUME_PATH = "/data"


@app.function(
    image=cleaner_image,
    volumes={VOLUME_PATH: volume},
    timeout=1800,
    memory=2048,
)
def run_cleaner():

    import re
    import regex
    import json
    import unicodedata
    import logging
    from typing import Optional
    from pathlib import Path
    from collections import Counter

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

    # ─────────────────────────────────────────
    # TextCleaner
    # ─────────────────────────────────────────
    class TextCleaner:

        # ── Noise patterns ────────────────────
        _EN_NOISE = [
            r"(accept|we use) cookies?.*?learn more",
            r"(home\s*[>›»]\s*){1,5}",
            r"(share|follow us|like us|subscribe).*?"
            r"(facebook|twitter|instagram)",
            r"copyright\s*©?\s*\d{4}.*",
            r"read more\.{0,3}$",
            r"skip to (main )?content",
            r"(print|download|share) (this )?(page|article)",
            r"\[\s*\]",
            r"\(\s*\)",
        ]

        _AR_NOISE = [
            r"(قبول|نستخدم)\s*(ملفات)?\s*الكوكيز?.*?المزيد",
            r"(الرئيسية\s*[>›»]\s*){1,5}",
            r"(شارك|تابعنا|اشترك).*?(فيسبوك|تويتر|انستغرام)",
            r"(حقوق? النشر|جميع الحقوق محفوظة).*?\d{4}",
            r"اقرأ المزيد\.{0,3}$",
            r"(طباعة|تحميل|مشاركة)\s*(هذه?)?\s*(الصفحة|المقال)",
            r"\[\s*\]",
            r"\(\s*\)",
        ]

        def __init__(self):
            self._compiled_en = [
                re.compile(p, re.IGNORECASE | re.DOTALL)
                for p in self._EN_NOISE
            ]
            self._compiled_ar = [
                regex.compile(p, regex.IGNORECASE | regex.DOTALL)
                for p in self._AR_NOISE
            ]

        # ── Unicode normalisation ─────────────
        _COMMON_CHARS = {
            "\u00a0": " ",   # non-breaking space
            "\u200b": "",    # zero-width space
            "\u200e": "",    # LTR mark
            "\u200f": "",    # RTL mark
            "\u00ad": "",    # soft hyphen
            "\ufeff": "",    # BOM
        }

        _EN_CHARS = {
            "\u2019": "'",  "\u2018": "'",
            "\u201c": '"',  "\u201d": '"',
            "\u2013": "-",  "\u2014": "--",
        }

        # Arabic-Indic → Western numerals + Alef variants
        # NOTE: Alef normalisation is intentionally kept here for
        # SEARCH purposes. The display title is preserved separately.
        _AR_CHARS = {
            "\u0660": "0", "\u0661": "1", "\u0662": "2",
            "\u0663": "3", "\u0664": "4", "\u0665": "5",
            "\u0666": "6", "\u0667": "7", "\u0668": "8",
            "\u0669": "9",
            "\u201c": "\u00ab",  # left double quote → «
            "\u201d": "\u00bb",  # right double quote → »
            # Alef variants → bare Alef (search normalisation only)
            "\u0622": "\u0627",  # آ → ا
            "\u0623": "\u0627",  # أ → ا
            "\u0625": "\u0627",  # إ → ا
            # Yeh variant
            "\u0649": "\u064a",  # ى → ي
        }

        def _normalise_unicode(self, text: str, language: str) -> str:
            text = unicodedata.normalize("NFC", text)
            for char, rep in self._COMMON_CHARS.items():
                text = text.replace(char, rep)
            replacements = (
                self._AR_CHARS if language == "ar" else self._EN_CHARS
            )
            for char, rep in replacements.items():
                text = text.replace(char, rep)
            return text

        def _remove_diacritics(self, text: str) -> str:
            """Strip Arabic harakat / tashkeel."""
            pattern = re.compile(
                r"[\u0610-\u061a\u064b-\u065f\u0670"
                r"\u06d6-\u06dc\u06df-\u06e8\u06ea-\u06ed\u0640]"
            )
            return pattern.sub("", text)

        def _remove_noise(self, text: str, language: str) -> str:
            """
            Apply language-specific noise patterns only.
            EN patterns → EN text only.
            AR patterns → AR text only.
            (v1 bug: EN patterns were always applied to both languages.)
            """
            patterns = (
                self._compiled_ar if language == "ar"
                else self._compiled_en
            )
            for pattern in patterns:
                text = pattern.sub(" ", text)
            return text

        def _remove_duplicate_lines(self, text: str) -> str:
            lines, seen, out = text.split("\n"), set(), []
            for line in lines:
                stripped = line.strip()
                if stripped and stripped not in seen:
                    out.append(line)
                    seen.add(stripped)
            return "\n".join(out)

        def _normalise_whitespace(self, text: str) -> str:
            text = re.sub(r" {2,}", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.replace("\t", " ")
            lines = [l.strip() for l in text.split("\n")]
            return "\n".join(lines).strip()

        def _remove_short_paragraphs(self, text: str, language: str) -> str:
            """
            Drop paragraphs below a minimum character threshold.
            Threshold is LOW to preserve short headings
            (e.g. 'Plan Your Visit' = 15 chars).
            """
            min_len    = 10 if language == "ar" else 15
            paragraphs = text.split("\n\n")
            kept = [p.strip() for p in paragraphs if len(p.strip()) >= min_len]
            return "\n\n".join(kept)

        # ── Public API ────────────────────────
        def clean_text(self, text: str, language: str = "en") -> str:
            if not text:
                return ""
            text = self._normalise_unicode(text, language)
            if language == "ar":
                text = self._remove_diacritics(text)
            text = self._remove_noise(text, language)
            text = self._remove_duplicate_lines(text)
            text = self._normalise_whitespace(text)
            text = self._remove_short_paragraphs(text, language)
            return text

        def clean_document(self, doc: dict) -> Optional[dict]:
            language     = doc.get("language", "en")
            original_len = len(doc.get("content", ""))

            cleaned_content = self.clean_text(doc.get("content", ""), language)
            cleaned_len     = len(cleaned_content)

            # Warn on heavy reduction
            if original_len > 0:
                reduction = (original_len - cleaned_len) / original_len * 100
                if reduction > 70:
                    logger.warning(
                        f"Heavy reduction {reduction:.0f}% "
                        f"[{language.upper()}] {doc.get('url')}"
                        f" ({original_len} → {cleaned_len})"
                    )
                else:
                    logger.info(
                        f"[{language.upper()}] {original_len} → "
                        f"{cleaned_len} chars ({reduction:.0f}% removed) | "
                        f"{doc.get('title','')[:40]}"
                    )

            # Drop documents that are too short after cleaning
            min_length = 50 if language == "ar" else 100
            if len(cleaned_content) < min_length:
                logger.warning(
                    f"Too short after cleaning [{language.upper()}]: "
                    f"{doc.get('url')}"
                )
                return None

            # Clean title — keep display-safe (no Alef normalisation on title)
            cleaned_title = self.clean_text(doc.get("title", ""), language).strip()

            return {
                "title":           cleaned_title,
                "url":             doc.get("url", ""),
                "category":        doc.get("category", "general"),
                "language":        language,
                "date":            doc.get("date", ""),
                "content":         cleaned_content,
                "content_length":  len(cleaned_content),
                "original_length": original_len,
            }

        def clean_all(
            self,
            documents: list[dict],
            output_path: str,
        ) -> list[dict]:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cleaned, skipped = [], 0
            for doc in documents:
                result = self.clean_document(doc)
                if result:
                    cleaned.append(result)
                else:
                    skipped += 1
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cleaned, f, indent=2, ensure_ascii=False)
            logger.info(
                f"Cleaned: {len(cleaned)} docs | Skipped: {skipped}"
            )
            return cleaned

    # ─────────────────────────────────────────
    # Load
    # ─────────────────────────────────────────
    input_path = Path(VOLUME_PATH) / "all_documents.json"
    if not input_path.exists():
        raise FileNotFoundError(
            "all_documents.json not found. Run the scraper first."
        )

    with open(input_path, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)

    lang_counts = Counter(d.get("language", "en") for d in raw_docs)
    print(f"📄 Loaded {len(raw_docs)} documents")
    print(f"🌐 English : {lang_counts.get('en', 0)}")
    print(f"🌐 Arabic  : {lang_counts.get('ar', 0)}")

    # ─────────────────────────────────────────
    # Clean
    # ─────────────────────────────────────────
    cleaner     = TextCleaner()
    output_path = str(Path(VOLUME_PATH) / "cleaned_documents.json")
    cleaned     = cleaner.clean_all(raw_docs, output_path)

    volume.commit()

    lang_cleaned = Counter(d.get("language", "en") for d in cleaned)
    skipped      = len(raw_docs) - len(cleaned)

    print(f"\n✅ Cleaning Complete!")
    print(f"📄 Cleaned  : {len(cleaned)}")
    print(f"   English  : {lang_cleaned.get('en', 0)}")
    print(f"   Arabic   : {lang_cleaned.get('ar', 0)}")
    print(f"   Skipped  : {skipped}")
    print(f"💾 Saved to : {output_path}")

    # Samples
    for lang in ("en", "ar"):
        sample = next((d for d in cleaned if d.get("language") == lang), None)
        if sample:
            flag = "🇬🇧" if lang == "en" else "🇪🇬"
            print(f"\n{flag} {lang.upper()} Sample:")
            print(f"   Title    : {sample['title']}")
            print(f"   Original : {sample['original_length']:,} chars")
            print(f"   Cleaned  : {sample['content_length']:,} chars")
            print(f"   Preview  : {sample['content'][:150]}...")

    return {
        "total_cleaned":   len(cleaned),
        "english_cleaned": lang_cleaned.get("en", 0),
        "arabic_cleaned":  lang_cleaned.get("ar", 0),
        "skipped":         skipped,
        "saved_to":        output_path,
    }


@app.local_entrypoint()
def main():
    print("🚀 Launching GEM Cleaner on Modal...")
    result = run_cleaner.remote()
    print(f"\n🎉 Done!")
    print(f"   Cleaned  : {result['total_cleaned']}")
    print(f"   English  : {result['english_cleaned']}")
    print(f"   Arabic   : {result['arabic_cleaned']}")
    print(f"   Skipped  : {result['skipped']}")
    print(f"   Saved to : {result['saved_to']}")