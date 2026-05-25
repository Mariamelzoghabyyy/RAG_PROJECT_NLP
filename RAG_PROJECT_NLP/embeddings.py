"""
gem_embedder.py

Converts text chunks into vector embeddings.
— Modal compatible  |  GPU accelerated (T4)
— Arabic + English bilingual
— Volume: "rag"
— Model: intfloat/multilingual-e5-base

Fixes over v1:
- Model cached in Modal volume (no re-download between runs)
- Embedding validation (NaN check, norm check)
- Unused 'language' param removed from prepare_text_for_embedding
- Batch size raised to 128 (T4 16GB can handle it)
- embed_query() exported for reuse by retriever
"""

import modal
from pathlib import Path

app = modal.App("gem-embedder")

embedder_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "sentence-transformers",
        "torch",
        "transformers",
        "numpy",
    ])
)

volume      = modal.Volume.from_name("rag", create_if_missing=True)
VOLUME_PATH = "/data"
MODEL_NAME  = "intfloat/multilingual-e5-base"


@app.function(
    image=embedder_image,
    volumes={VOLUME_PATH: volume},
    gpu="T4",
    timeout=3600,
    memory=16384,
)
def run_embedder(
    batch_size:    int  = 128,    # T4 16GB handles 128 comfortably
    show_progress: bool = True,
):
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

    # ─────────────────────────────────────────
    # Helpers (inlined for Modal)
    # ─────────────────────────────────────────
    def e5_passage(text: str) -> str:
        """multilingual-e5 REQUIRES 'passage: ' prefix on documents."""
        return f"passage: {text}"

    def e5_query(text: str) -> str:
        """multilingual-e5 REQUIRES 'query: ' prefix on queries."""
        return f"query: {text}"

    def validate_embeddings(
        embeddings: np.ndarray,
        logger,
    ) -> bool:
        """Sanity-check embedding matrix before saving."""
        if np.isnan(embeddings).any():
            logger.error("❌ NaN values detected in embeddings!")
            return False
        norms = np.linalg.norm(embeddings, axis=1)
        bad   = np.where((norms < 0.98) | (norms > 1.02))[0]
        if len(bad) > 0:
            logger.warning(
                f"⚠️  {len(bad)} embeddings not unit-normalised "
                f"(norms range: {norms.min():.4f}–{norms.max():.4f})"
            )
        else:
            logger.info("✅ All embeddings are unit-normalised")
        return True

    # ─────────────────────────────────────────
    # EmbeddingGenerator
    # ─────────────────────────────────────────
    class EmbeddingGenerator:
        """
        Multilingual embedding generator using intfloat/multilingual-e5-base.

        Why this model:
        ─────────────────────────────────────────
        - Trained on 100+ languages including Arabic
        - Requires 'passage: ' / 'query: ' prefixes (enforced here)
        - 768-dim embeddings — good quality/speed trade-off
        - Cross-lingual: Arabic query finds English chunks and vice versa
        - FREE — no API key needed

        Model is cached in the Modal volume to avoid re-downloading
        on every run (saves ~1 min per invocation).
        """

        def __init__(self, model_name: str = MODEL_NAME):
            self.model_name    = model_name
            self.device        = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_cache   = str(Path(VOLUME_PATH) / "model_cache")

            # Cache model in volume — skip download on subsequent runs
            Path(self.model_cache).mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Loading {model_name} on {self.device} "
                f"(cache: {self.model_cache})"
            )
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                cache_folder=self.model_cache,
            )
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dim: {self.embedding_dim}")

        def generate_embeddings(
            self,
            chunks: list[dict],
            batch_size: int = 128,
            show_progress: bool = True,
        ) -> np.ndarray:
            """
            Generate L2-normalised embeddings for all chunks.
            Arabic + English are processed in the same batch.
            Returns ndarray of shape (N, embedding_dim).
            """
            lang_counts = Counter(c.get("language", "en") for c in chunks)
            logger.info(
                f"Embedding {len(chunks)} chunks "
                f"(EN: {lang_counts.get('en',0)}, "
                f"AR: {lang_counts.get('ar',0)}) "
                f"on {self.device}"
            )

            # Apply e5 passage prefix — same for all languages
            texts = [e5_passage(c.get("content", "")) for c in chunks]

            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,   # L2 normalise → cosine via dot product
                device=self.device,
            )

            validate_embeddings(embeddings, logger)
            return embeddings

        def embed_query(self, query: str) -> np.ndarray:
            """
            Embed a single user query.
            Applies 'query: ' prefix as required by multilingual-e5.
            Works for Arabic and English without any language detection.
            """
            embedding = self.model.encode(
                e5_query(query),
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device,
            )
            return embedding

    # ─────────────────────────────────────────
    # Load chunks
    # ─────────────────────────────────────────
    chunks_path = Path(VOLUME_PATH) / "all_chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(
            "all_chunks.json not found. Run the chunker first."
        )

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    lang_counts = Counter(c.get("language", "en") for c in chunks)
    print(f"📦 Loaded {len(chunks)} chunks")
    print(f"🌐 English : {lang_counts.get('en', 0)}")
    print(f"🌐 Arabic  : {lang_counts.get('ar', 0)}")

    # ─────────────────────────────────────────
    # Embed
    # ─────────────────────────────────────────
    generator  = EmbeddingGenerator(MODEL_NAME)
    embeddings = generator.generate_embeddings(
        chunks, batch_size=batch_size, show_progress=show_progress
    )

    # ─────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────
    output_path = str(Path(VOLUME_PATH) / "embeddings.npy")
    np.save(output_path, embeddings)

    volume.commit()

    size_mb = embeddings.nbytes / 1024 / 1024
    print(f"\n✅ Embedding Complete!")
    print(f"📐 Shape      : {embeddings.shape}")
    print(f"📏 Dim        : {generator.embedding_dim}")
    print(f"🤖 Model      : {generator.model_name}")
    print(f"⚡ Device     : {generator.device}")
    print(f"💾 Size       : {size_mb:.2f} MB")
    print(f"💾 Saved to   : {output_path}")

    return {
        "total_chunks":    len(chunks),
        "english_chunks":  lang_counts.get("en", 0),
        "arabic_chunks":   lang_counts.get("ar", 0),
        "embedding_shape": list(embeddings.shape),
        "embedding_dim":   generator.embedding_dim,
        "model":           MODEL_NAME,
        "size_mb":         round(size_mb, 2),
        "saved_to":        output_path,
    }


@app.local_entrypoint()
def main():
    print("🚀 Launching GEM Embedder on Modal (GPU T4)...")
    result = run_embedder.remote(batch_size=128, show_progress=True)
    print(f"\n🎉 Done!")
    print(f"   Chunks  : {result['total_chunks']}")
    print(f"   Shape   : {result['embedding_shape']}")
    print(f"   Dim     : {result['embedding_dim']}")
    print(f"   Size    : {result['size_mb']} MB")
    print(f"   Saved   : {result['saved_to']}")