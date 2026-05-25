"""
gem_retriever.py

Semantic search + chunk retrieval for the GEM RAG pipeline.
— Modal compatible
— Arabic + English bilingual
— Volume: "rag"
— Model: intfloat/multilingual-e5-base

Fixes over v1:
- GPU removed (CPU is sufficient for single-query retrieval)
- score_threshold default raised to 0.45 (reduces noise)
- Dynamic threshold floor: max(threshold, best_score * 0.7)
- category_filter uses larger search_k to avoid empty results
- Arabic query.lower() removed (Arabic has no case)
- EmbeddingGenerator no longer duplicated — single implementation
- Model loaded from volume cache
- chunks_metadata loaded from JSON (no pickle)
"""

import modal
from pathlib import Path

app = modal.App("gem-retriever")

retriever_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "sentence-transformers",
        "torch",
        "transformers",
        "numpy",
        "faiss-cpu",
    ])
)

volume      = modal.Volume.from_name("rag", create_if_missing=True)
VOLUME_PATH = "/data"
MODEL_NAME  = "intfloat/multilingual-e5-base"


@app.function(
    image=retriever_image,
    volumes={VOLUME_PATH: volume},
    # No GPU — single-query embedding + flat FAISS search is fast on CPU
    timeout=600,
    memory=8192,
)
def retrieve(
    query:           str,
    top_k:           int   = 5,
    score_threshold: float = 0.45,   # raised from 0.3 — reduces noise
    language_filter: str   = None,
    category_filter: str   = None,
) -> dict:
    """
    Retrieve the top-k most relevant chunks for a query.

    Args:
        query            : User question in Arabic or English
        top_k            : Chunks to return
        score_threshold  : Minimum cosine similarity (0–1)
        language_filter  : None | "ar" | "en"
        category_filter  : None | "collection" | "visitor_info" | etc.

    Returns:
        {query, query_lang, query_type, chunks, context, sources}
    """

    import json
    import logging
    import numpy as np
    import faiss
    from pathlib import Path
    from collections import Counter
    from typing import Optional

    from sentence_transformers import SentenceTransformer
    import torch

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

    def detect_language(text: str) -> str:
        return "ar" if is_arabic_text(text) else "en"

    def e5_query(text: str) -> str:
        return f"query: {text}"

    # ─────────────────────────────────────────
    # QueryClassifier
    # ─────────────────────────────────────────
    class QueryClassifier:
        """
        Classifies queries into types for prompt template selection.
        Uses keyword matching — fast and reliable for museum queries.
        Types: visitor_info | history | general
        """

        _VISITOR_EN = [
            "open", "opening", "hours", "ticket", "price", "cost",
            "how much", "parking", "entrance", "location", "address",
            "how to get", "transport", "bus", "metro", "taxi",
            "accessibility", "wheelchair", "disabled", "facilities",
            "restaurant", "cafe", "shop", "gift", "tour", "guide",
            "visit", "plan", "tips", "rules", "allowed", "prohibited",
            "photography", "camera", "bag", "locker", "children",
        ]

        _VISITOR_AR = [
            "مواعيد", "ساعات", "فتح", "تذاكر", "سعر", "تكلفة",
            "كم", "موقع", "عنوان", "وصول", "مواصلات", "أتوبيس",
            "مترو", "تاكسي", "إتاحة", "ذوي", "مرافق", "مطعم",
            "كافيه", "متجر", "هدايا", "جولة", "دليل", "زيارة",
            "خطط", "نصائح", "قواعد", "مسموح", "محظور", "تصوير",
            "كاميرا", "حقيبة", "خزانة", "أطفال", "دخول",
        ]

        _HISTORY_EN = [
            "history", "historical", "ancient", "pharaoh", "dynasty",
            "kingdom", "artifact", "artefact", "statue", "mask",
            "tomb", "mummy", "hieroglyph", "ramesses", "ramses",
            "tutankhamun", "cleopatra", "ptolemaic", "obelisk",
            "temple", "pyramid", "sphinx", "mythology", "god", "goddess",
            "hathor", "sekhmet", "ptah", "osiris", "isis", "horus",
            "BC", "BCE", "century", "era", "period", "civilization",
            "discovery", "excavation", "archaeologist",
        ]

        _HISTORY_AR = [
            "تاريخ", "تاريخي", "قديم", "فرعون", "أسرة", "مملكة",
            "مصر", "أثر", "تمثال", "قناع", "مقبرة", "مومياء",
            "هيروغليف", "رمسيس", "توت", "كليوباترا", "بطلمي",
            "مسلة", "معبد", "هرم", "أبو الهول", "أسطورة",
            "إله", "إلهة", "حتحور", "سخمت", "بتاح", "أوزير",
            "إيزيس", "حورس", "قبل الميلاد", "قرن", "عصر", "حضارة",
            "اكتشاف", "حفريات", "آثاري",
        ]

        def classify(self, query: str, language: str) -> str:
            # Arabic has no case — skip .lower() for AR
            q = query if language == "ar" else query.lower()

            visitor_kws = self._VISITOR_AR if language == "ar" else self._VISITOR_EN
            history_kws = self._HISTORY_AR if language == "ar" else self._HISTORY_EN

            visitor_hits = sum(1 for kw in visitor_kws if kw in q)
            history_hits = sum(1 for kw in history_kws if kw in q)

            if visitor_hits == 0 and history_hits == 0:
                return "general"
            return "visitor_info" if visitor_hits >= history_hits else "history"

    # ─────────────────────────────────────────
    # EmbeddingGenerator
    # ─────────────────────────────────────────
    class EmbeddingGenerator:

        def __init__(self):
            self.device      = "cuda" if torch.cuda.is_available() else "cpu"
            model_cache      = str(Path(VOLUME_PATH) / "model_cache")
            Path(model_cache).mkdir(exist_ok=True)
            logger.info(f"Loading model on {self.device}")
            self.model       = SentenceTransformer(
                MODEL_NAME,
                device=self.device,
                cache_folder=model_cache,
            )
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

        def embed(self, query: str) -> np.ndarray:
            return self.model.encode(
                e5_query(query),
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device,
            )

    # ─────────────────────────────────────────
    # FAISSSearcher
    # ─────────────────────────────────────────
    class FAISSSearcher:

        def __init__(self):
            self.index  = None
            self.chunks = []

        def load(self, index_dir: str):
            self.index = faiss.read_index(f"{index_dir}/faiss_index.bin")
            with open(f"{index_dir}/chunks_metadata.json", "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            logger.info(
                f"✅ Loaded FAISS: {self.index.ntotal} vectors, "
                f"{len(self.chunks)} chunks"
            )

        def search(
            self,
            query_embedding:  np.ndarray,
            top_k:            int,
            score_threshold:  float,
            language_filter:  Optional[str],
            category_filter:  Optional[str],
        ) -> list[dict]:
            query = query_embedding.astype(np.float32)
            if query.ndim == 1:
                query = query.reshape(1, -1)

            # Retrieve more candidates when filters are active
            # to avoid empty results after filtering
            filter_active = bool(language_filter or category_filter)
            search_k      = top_k * 10 if filter_active else top_k * 2

            scores, indices = self.index.search(query, search_k)

            # Dynamic threshold: floor at score_threshold,
            # but also require results to be at least 70% of best score
            if len(indices[0]) > 0 and indices[0][0] != -1:
                best_score      = float(scores[0][0])
                dynamic_floor   = max(score_threshold, best_score * 0.70)
            else:
                dynamic_floor   = score_threshold

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or score < dynamic_floor:
                    continue

                chunk = self.chunks[idx].copy()
                chunk["similarity_score"] = float(score)

                if language_filter and chunk.get("language") != language_filter:
                    continue
                if category_filter and chunk.get("category") != category_filter:
                    continue

                results.append(chunk)
                if len(results) >= top_k:
                    break

            return results

    # ─────────────────────────────────────────
    # Retriever orchestration
    # ─────────────────────────────────────────
    embedder   = EmbeddingGenerator()
    searcher   = FAISSSearcher()
    classifier = QueryClassifier()

    searcher.load(VOLUME_PATH)

    # 1. Detect language
    query_lang = detect_language(query)

    # 2. Classify query type
    query_type = classifier.classify(query, query_lang)

    logger.info(
        f"[{query_lang.upper()}] [{query_type}] '{query[:80]}'"
    )

    # 3. Embed query
    query_embedding = embedder.embed(query)

    # 4. Search
    results = searcher.search(
        query_embedding=query_embedding,
        top_k=top_k,
        score_threshold=score_threshold,
        language_filter=language_filter,
        category_filter=category_filter,
    )

    # 5. Quality warning
    if results:
        avg_score = sum(r["similarity_score"] for r in results) / len(results)
        if avg_score < 0.5:
            logger.warning(
                f"Low avg similarity {avg_score:.3f} for '{query[:60]}'"
            )

    # 6. No results
    if not results:
        no_result = (
            "لم يتم العثور على معلومات ذات صلة في قاعدة المعرفة."
            if query_lang == "ar"
            else "No relevant information found in the knowledge base."
        )
        return {
            "query":      query,
            "query_lang": query_lang,
            "query_type": query_type,
            "chunks":     [],
            "context":    no_result,
            "sources":    [],
        }

    # 7. Format context + sources
    context_parts = []
    sources       = []

    for i, chunk in enumerate(results, 1):
        c_lang  = chunk.get("language", "en")
        label   = f"[المصدر {i}]" if c_lang == "ar" else f"[Source {i}]"
        context_parts.append(
            f"{label}: {chunk['title']}\n{chunk['content']}"
        )
        sources.append({
            "title":    chunk["title"],
            "url":      chunk["url"],
            "category": chunk["category"],
            "language": c_lang,
            "score":    round(chunk["similarity_score"], 3),
        })

    return {
        "query":      query,
        "query_lang": query_lang,
        "query_type": query_type,
        "chunks":     results,
        "context":    "\n\n---\n\n".join(context_parts),
        "sources":    sources,
    }


# ─────────────────────────────────────────────
# Test Function
# ─────────────────────────────────────────────
@app.function(
    image=retriever_image,
    volumes={VOLUME_PATH: volume},
    timeout=600,
    memory=8192,
)
def run_retriever_test():
    """Test retrieval with 10 cross-lingual queries."""
    test_queries = [
        "What are the opening hours of the Grand Egyptian Museum?",
        "Tell me about Tutankhamun's treasures",
        "How can I buy tickets to the GEM?",
        "What is the history of Ramesses II?",
        "Are there educational programs for schools?",
        "ما هي مواعيد فتح المتحف المصري الكبير؟",
        "أخبرني عن كنوز توت عنخ آمون",
        "كيف يمكنني شراء تذاكر للمتحف؟",
        "ما تاريخ رمسيس الثاني؟",
        "هل توجد برامج تعليمية للمدارس؟",
    ]

    print(f"\n🔍 Retriever Test — {len(test_queries)} queries")
    print("=" * 60)

    for query in test_queries:
        result = retrieve.local(query=query, top_k=3)
        print(
            f"\n❓ [{result['query_lang'].upper()}] "
            f"[{result['query_type']}] {query[:70]}"
        )
        if not result["sources"]:
            print("  ⚠️  No results")
            continue
        for s in result["sources"]:
            print(
                f"  → [{s['language'].upper()}] "
                f"[{s['score']:.3f}] {s['title'][:50]}"
            )

    return {"queries_tested": len(test_queries)}


@app.local_entrypoint()
def main():
    print("🚀 Launching GEM Retriever Test on Modal...")
    run_retriever_test.remote()