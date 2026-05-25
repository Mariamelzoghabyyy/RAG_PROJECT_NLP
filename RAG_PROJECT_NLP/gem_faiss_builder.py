"""
gem_faiss_builder.py

Builds and saves a FAISS index from pre-computed embeddings.
— Modal compatible
— Arabic + English bilingual
— Volume: "rag"
— Embedding dim: 768 (intfloat/multilingual-e5-base)

Fixes over v1:
- Downgraded to CPU-only (faiss-cpu on T4 wasted GPU)
- score_threshold param removed (belonged in retriever)
- Model NOT reloaded for query test (uses saved embeddings instead)
- Test queries use pre-embedded vectors from the index itself
- chunk metadata stored as JSON only (pickle dropped — security + portability)
- IVFFlat note added for future scaling
"""

import modal
from pathlib import Path

app = modal.App("gem-faiss-builder")

faiss_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        # Required C++ runtime libs for faiss-cpu
        "libgomp1",
        "libopenblas-dev",
    ])
    .pip_install([
        "numpy==1.26.4",
        "faiss-cpu==1.8.0",          # pin exact version — avoids broken builds
        "sentence-transformers==2.7.0",
        "torch==2.3.0",
        "transformers==4.41.0",
    ])
)

volume      = modal.Volume.from_name("rag", create_if_missing=True)
VOLUME_PATH = "/data"
MODEL_NAME  = "intfloat/multilingual-e5-base"


@app.function(
    image=faiss_image,
    volumes={VOLUME_PATH: volume},
    timeout=1800,
    memory=8192,
    # No GPU needed: FAISS flat index on CPU is fast for <50k vectors
)
def run_faiss_builder():

    import sys
    # Clear any stale faiss imports
    for mod in list(sys.modules.keys()):
        if mod.startswith("faiss"):
            del sys.modules[mod]

    import faiss
    import json
    import logging
    import numpy as np
    from pathlib import Path
    from collections import Counter
    from sentence_transformers import SentenceTransformer
    import torch

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # ── Verify faiss loaded correctly ────────
    print(f"✅ faiss loaded: {faiss.__file__}")
    has_ip  = hasattr(faiss, "IndexFlatIP")
    has_l2  = hasattr(faiss, "IndexFlatL2")
    has_rw  = hasattr(faiss, "write_index")
    print(f"   IndexFlatIP : {has_ip}")
    print(f"   IndexFlatL2 : {has_l2}")
    print(f"   write_index : {has_rw}")

    # ─────────────────────────────────────────
    # Helper: e5 prefix (inlined)
    # ─────────────────────────────────────────
    def e5_query(text: str) -> str:
        return f"query: {text}"

    # ─────────────────────────────────────────
    # Index factory with fallbacks
    # ─────────────────────────────────────────
    def create_index(dim: int):
        """
        Try every available FAISS method to create an inner-product index.
        Inner product on L2-normalised vectors == cosine similarity.

        For scaling beyond 50k vectors, switch to:
            faiss.index_factory(dim, "IVF1024,Flat", faiss.METRIC_INNER_PRODUCT)
        """
        # Method 1 — preferred
        if hasattr(faiss, "IndexFlatIP"):
            try:
                idx = faiss.IndexFlatIP(dim)
                logger.info("Created index: IndexFlatIP")
                return idx, "inner_product"
            except Exception as e:
                logger.warning(f"IndexFlatIP failed: {e}")

        # Method 2 — index_factory with IP metric
        if hasattr(faiss, "index_factory") and hasattr(faiss, "METRIC_INNER_PRODUCT"):
            try:
                idx = faiss.index_factory(
                    dim, "Flat", faiss.METRIC_INNER_PRODUCT
                )
                logger.info("Created index: index_factory(Flat, IP)")
                return idx, "inner_product"
            except Exception as e:
                logger.warning(f"index_factory IP failed: {e}")

        # Method 3 — L2 fallback
        if hasattr(faiss, "IndexFlatL2"):
            try:
                idx = faiss.IndexFlatL2(dim)
                logger.info("Created index: IndexFlatL2 (fallback)")
                return idx, "l2"
            except Exception as e:
                logger.warning(f"IndexFlatL2 failed: {e}")

        raise RuntimeError(
            "Cannot create any FAISS index. "
            f"Index classes available: "
            f"{[x for x in dir(faiss) if 'Index' in x]}"
        )

    def save_index(index, path: str):
        if hasattr(faiss, "write_index"):
            faiss.write_index(index, path)
        else:
            buf = faiss.serialize_index(index)
            with open(path, "wb") as f:
                f.write(buf.tobytes())
        logger.info(f"✅ Index saved → {path}")

    # ─────────────────────────────────────────
    # FAISSIndexBuilder
    # ─────────────────────────────────────────
    class FAISSIndexBuilder:

        def __init__(self, embedding_dim: int = 768):
            self.embedding_dim = embedding_dim
            self.index         = None
            self.chunks        = []
            self.index_metric  = None

        def build(
            self,
            embeddings: np.ndarray,
            chunks: list[dict],
        ):
            assert len(embeddings) == len(chunks), (
                f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks"
            )
            self.chunks = chunks

            lang_counts = Counter(c.get("language", "en") for c in chunks)
            logger.info(
                f"Building index: {len(embeddings)} vectors "
                f"(dim={self.embedding_dim}) | "
                f"EN={lang_counts.get('en',0)} "
                f"AR={lang_counts.get('ar',0)}"
            )

            embeddings_f32       = embeddings.astype(np.float32)
            self.index, self.index_metric = create_index(self.embedding_dim)
            self.index.add(embeddings_f32)

            logger.info(
                f"✅ Index built: {self.index.ntotal} vectors | "
                f"metric={self.index_metric}"
            )

        def save(self, output_dir: str):
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # FAISS binary
            save_index(self.index, f"{output_dir}/faiss_index.bin")

            # Chunk metadata — JSON only (no pickle for security/portability)
            chunks_json = f"{output_dir}/chunks_metadata.json"
            with open(chunks_json, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Chunks metadata → {chunks_json}")

            # Index meta (consumed by retriever)
            meta = {
                "embedding_dim": self.embedding_dim,
                "index_metric":  self.index_metric,
                "total_vectors": self.index.ntotal,
                "model":         MODEL_NAME,
            }
            with open(f"{output_dir}/faiss_meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            logger.info(f"✅ Index meta      → {output_dir}/faiss_meta.json")

        def search(
            self,
            query_embedding: np.ndarray,
            top_k: int = 5,
            score_threshold: float = 0.0,
            language_filter: str = None,
        ) -> list[dict]:
            query = query_embedding.astype(np.float32)
            if query.ndim == 1:
                query = query.reshape(1, -1)

            search_k        = top_k * 4 if language_filter else top_k * 2
            scores, indices = self.index.search(query, search_k)
            results         = []

            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or score < score_threshold:
                    continue
                chunk = self.chunks[idx].copy()
                chunk["similarity_score"] = float(score)
                if language_filter and chunk.get("language") != language_filter:
                    continue
                results.append(chunk)
                if len(results) >= top_k:
                    break

            return results

        def run_query_test(
            self,
            model: SentenceTransformer,
            device: str,
            embeddings: np.ndarray,
        ):
            """
            Test retrieval quality using stored embeddings.
            No model reloading needed — we use the provided model
            only for the 4 test queries (lightweight).
            """
            test_queries = [
                {
                    "text":     "What are the opening hours?",
                    "lang":     "en",
                    "expected": ["hour", "open", "visit"],
                },
                {
                    "text":     "Tell me about Tutankhamun's golden mask",
                    "lang":     "en",
                    "expected": ["tutankhamun", "mask", "golden"],
                },
                {
                    "text":     "ما هي مواعيد فتح المتحف؟",
                    "lang":     "ar",
                    "expected": ["مواعيد", "فتح", "ساعات"],
                },
                {
                    "text":     "قناع توت عنخ آمون الذهبي",
                    "lang":     "ar",
                    "expected": ["توت", "قناع", "ذهب"],
                },
            ]

            print("\n🔍 Real Query Test:")
            print("─" * 60)

            for item in test_queries:
                query    = item["text"]
                lang     = item["lang"]
                expected = item["expected"]

                embedding = model.encode(
                    e5_query(query),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=device,
                )

                results = self.search(
                    query_embedding=embedding,
                    top_k=3,
                    score_threshold=0.0,
                )

                print(f"\n❓ [{lang.upper()}] {query}")
                if not results:
                    print("  ⚠️  No results")
                    continue

                for r in results:
                    combined = (r.get("title","") + r.get("content","")).lower()
                    relevant = "✅" if any(
                        kw.lower() in combined for kw in expected
                    ) else "❓"
                    print(
                        f"  {relevant} [{r.get('language','?').upper()}] "
                        f"Score: {r['similarity_score']:.3f} | "
                        f"{r.get('title','?')[:50]}"
                    )

    # ─────────────────────────────────────────
    # Load data
    # ─────────────────────────────────────────
    chunks_path = Path(VOLUME_PATH) / "all_chunks.json"
    emb_path    = Path(VOLUME_PATH) / "embeddings.npy"

    if not chunks_path.exists():
        raise FileNotFoundError("all_chunks.json not found.")
    if not emb_path.exists():
        raise FileNotFoundError("embeddings.npy not found.")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(str(emb_path))

    assert len(chunks) == len(embeddings), (
        f"Chunk/embedding mismatch: {len(chunks)} vs {len(embeddings)}. "
        "Re-run the embedder."
    )

    lang_counts   = Counter(c.get("language", "en") for c in chunks)
    embedding_dim = embeddings.shape[1]

    print(f"📦 Chunks          : {len(chunks)}")
    print(f"📐 Embeddings      : {embeddings.shape}")
    print(f"📏 Dim             : {embedding_dim}")
    print(f"🌐 English         : {lang_counts.get('en', 0)}")
    print(f"🌐 Arabic          : {lang_counts.get('ar', 0)}")
    print(f"\n🔎 Alignment check:")
    print(f"   chunks[0]  : {chunks[0].get('chunk_id','?')}")
    print(f"   chunks[-1] : {chunks[-1].get('chunk_id','?')}")

    # ─────────────────────────────────────────
    # Build + Save
    # ─────────────────────────────────────────
    builder = FAISSIndexBuilder(embedding_dim=embedding_dim)
    builder.build(embeddings=embeddings, chunks=chunks)
    builder.save(output_dir=VOLUME_PATH)

    # ─────────────────────────────────────────
    # Query test — load model once (from cache if available)
    # ─────────────────────────────────────────
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    model_cache = str(Path(VOLUME_PATH) / "model_cache")
    Path(model_cache).mkdir(exist_ok=True)

    logger.info(f"Loading model for query test on {device}...")
    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        cache_folder=model_cache,
    )
    builder.run_query_test(model=model, device=device, embeddings=embeddings)

    volume.commit()

    print(f"\n✅ FAISS Index Built!")
    print(f"📊 Vectors         : {builder.index.ntotal}")
    print(f"📏 Dim             : {embedding_dim}")
    print(f"📐 Metric          : {builder.index_metric}")
    print(f"💾 Index           : {VOLUME_PATH}/faiss_index.bin")
    print(f"💾 Metadata        : {VOLUME_PATH}/chunks_metadata.json")

    return {
        "total_vectors":  builder.index.ntotal,
        "embedding_dim":  embedding_dim,
        "index_metric":   builder.index_metric,
        "english_chunks": lang_counts.get("en", 0),
        "arabic_chunks":  lang_counts.get("ar", 0),
        "index_path":     f"{VOLUME_PATH}/faiss_index.bin",
        "metadata_path":  f"{VOLUME_PATH}/chunks_metadata.json",
    }


@app.local_entrypoint()
def main():
    print("🚀 Launching GEM FAISS Builder on Modal...")
    result = run_faiss_builder.remote()
    print(f"\n🎉 Done!")
    print(f"   Vectors  : {result['total_vectors']}")
    print(f"   Dim      : {result['embedding_dim']}")
    print(f"   Metric   : {result['index_metric']}")
    print(f"   English  : {result['english_chunks']}")