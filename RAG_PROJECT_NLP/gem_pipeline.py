"""
gem_pipeline.py

Full RAG Pipeline as a Modal Web Endpoint (FastAPI).
— Serves HTTP requests from Streamlit or any client
— Arabic + English bilingual
— Volume: "rag"
— Model: intfloat/multilingual-e5-base + Groq Llama 3.1

Improvements over v2:
- Score threshold lowered to 0.30 (more chunks retrieved)
- Dynamic floor multiplier lowered to 0.60 (less aggressive filtering)
- Temperature raised to 0.7 (more generative, fluid responses)
- Prompts updated to allow expansion beyond sources
- Chat history properly formatted and passed to LLM
- Multi-turn conversation memory (last 6 turns)
- Step-by-step reasoning in prompts
- Clarifying question support for vague queries
"""

import modal
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List

# ─────────────────────────────────────────────
# Modal App & Infrastructure
# ─────────────────────────────────────────────
app = modal.App("gem-pipeline")

pipeline_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "libgomp1",
        "libopenblas-dev",
    ])
    .pip_install([
        "numpy==1.26.4",
        "faiss-cpu==1.8.0",
        "sentence-transformers==2.7.0",
        "torch==2.3.0",
        "transformers==4.41.0",
        "openai",
        "tiktoken",
        "fastapi",
        "pydantic",
    ])
)

volume      = modal.Volume.from_name("rag", create_if_missing=True)
VOLUME_PATH = "/data"
MODEL_NAME  = "intfloat/multilingual-e5-base"


# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────
class AskRequest(BaseModel):
    question:        str
    top_k:           int   = 5
    score_threshold: float = 0.30        # lowered from 0.45
    language_filter: Optional[str] = None
    category_filter: Optional[str] = None
    chat_history:    str   = ""          # formatted previous turns


class SourceModel(BaseModel):
    title:    str
    url:      str
    category: str
    language: str
    score:    float


class AskResponse(BaseModel):
    question:         str
    answer:           str
    sources:          List[SourceModel]
    query_type:       str
    query_language:   str
    chunks_retrieved: int


# ─────────────────────────────────────────────
# Core Processing Function (Runs in Cloud)
# ─────────────────────────────────────────────
@app.function(
    image=pipeline_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("gem-secrets")],
    timeout=300,
    memory=8192,
    min_containers=1,
)
def process_rag(request: AskRequest) -> AskResponse:
    """
    Core ML processing logic — runs entirely in Modal cloud.

    Flow:
      1. detect_language(question)
      2. classify_query → visitor_info | history | general
      3. embed query (multilingual-e5, 'query: ' prefix)
      4. FAISS search → top-k chunks
      5. dynamic threshold filtering
      6. truncate_context (protect LLM window)
      7. select prompt (AR/EN × query_type) — generative style
      8. Groq/OpenAI LLM → answer
      9. return answer + sources
    """
    import os
    import json
    import logging
    import numpy as np
    import faiss
    import tiktoken
    from pathlib import Path

    from sentence_transformers import SentenceTransformer
    from openai import OpenAI
    import torch

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # ─────────────────────────────────────────
    # Helpers
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

    def truncate_context(context: str, max_tokens: int = 3000) -> str:
        try:
            enc    = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(context)
            if len(tokens) <= max_tokens:
                return context
            truncated = enc.decode(tokens[:max_tokens])
            note = (
                "\n\n[تم اقتصاص السياق.]"
                if detect_language(context) == "ar"
                else "\n\n[Context truncated.]"
            )
            return truncated + note
        except Exception:
            return context[:max_tokens * 4]

    # ─────────────────────────────────────────
    # Query Classifier
    # ─────────────────────────────────────────
    class QueryClassifier:
        _VISITOR_EN = [
            "open", "opening", "hours", "ticket", "price", "cost",
            "how much", "parking", "entrance", "location", "address",
            "transport", "bus", "metro", "taxi", "accessibility",
            "wheelchair", "restaurant", "cafe", "shop", "gift",
            "tour", "guide", "visit", "plan", "tips", "rules",
            "photography", "camera", "locker", "children", "disabled",
        ]
        _VISITOR_AR = [
            "مواعيد", "ساعات", "فتح", "تذاكر", "سعر", "تكلفة",
            "كم", "موقع", "عنوان", "وصول", "مواصلات", "أتوبيس",
            "مترو", "تاكسي", "إتاحة", "ذوي", "مرافق", "مطعم",
            "كافيه", "متجر", "هدايا", "جولة", "دليل", "زيارة",
            "نصائح", "تصوير", "كاميرا", "خزانة", "أطفال", "دخول",
        ]
        _HISTORY_EN = [
            "history", "historical", "ancient", "pharaoh", "dynasty",
            "kingdom", "artifact", "artefact", "statue", "mask",
            "tomb", "mummy", "hieroglyph", "ramesses", "ramses",
            "tutankhamun", "cleopatra", "ptolemaic", "obelisk",
            "temple", "pyramid", "sphinx", "god", "goddess",
            "hathor", "sekhmet", "ptah", "osiris", "isis", "horus",
            "BC", "BCE", "century", "civilization", "excavation",
        ]
        _HISTORY_AR = [
            "تاريخ", "تاريخي", "قديم", "فرعون", "أسرة", "مملكة",
            "أثر", "تمثال", "قناع", "مقبرة", "مومياء", "هيروغليف",
            "رمسيس", "توت", "كليوباترا", "بطلمي", "مسلة", "معبد",
            "هرم", "أبو الهول", "إله", "إلهة", "حتحور", "سخمت",
            "بتاح", "أوزير", "إيزيس", "حورس", "قبل الميلاد", "حضارة",
        ]

        def classify(self, query: str, language: str) -> str:
            q     = query if language == "ar" else query.lower()
            v_kws = self._VISITOR_AR if language == "ar" else self._VISITOR_EN
            h_kws = self._HISTORY_AR if language == "ar" else self._HISTORY_EN
            v_hits = sum(1 for kw in v_kws if kw in q)
            h_hits = sum(1 for kw in h_kws if kw in q)
            if v_hits == 0 and h_hits == 0:
                return "general"
            return "visitor_info" if v_hits >= h_hits else "history"

    # ─────────────────────────────────────────
    # Generative Prompt Templates
    # ─────────────────────────────────────────
    _SYSTEM = {
        "en": (
            "You are an expert, friendly AI guide for the Grand Egyptian Museum (GEM) "
            "— the world's largest archaeological museum in Giza, Egypt. "
            "You are knowledgeable about ancient Egyptian history, the museum's "
            "collections, visitor services, and educational programs. "
            "Give rich, engaging, informative answers. "
            "Never invent specific facts like prices, opening hours, or dates "
            "that are not in your sources."
        ),
        "ar": (
            "أنت مرشد ذكاء اصطناعي خبير وودود للمتحف المصري الكبير (GEM) "
            "— أكبر متحف أثري في العالم في الجيزة، مصر. "
            "أنت متخصص في التاريخ المصري القديم ومجموعات المتحف "
            "وخدمات الزوار والبرامج التعليمية. "
            "قدم إجابات غنية وجذابة ومفيدة. "
            "لا تخترع أبداً حقائق محددة كالأسعار أو مواعيد العمل أو التواريخ "
            "التي لا توجد في مصادرك."
        ),
    }

    # ── English Prompts ───────────────────────
    _GENERAL_EN = """\
You are an expert AI guide for the Grand Egyptian Museum (GEM).

Use the provided museum sources as your PRIMARY reference.
You may enrich your answer with relevant Egyptology knowledge to give
a more engaging and complete response — but always prioritize the
museum's own content and never invent specific facts.

If the question is vague or unclear, ask a friendly clarifying question.

Think step by step:
1. What is the visitor really asking?
2. What do the sources tell us?
3. What additional context would make this answer richer?
4. Give a complete, engaging, friendly answer.

MUSEUM SOURCES:
{context}

CONVERSATION HISTORY:
{chat_history}

VISITOR'S QUESTION: {question}

ANSWER:"""

    _HISTORY_EN = """\
You are a passionate Egyptologist and historian at the Grand Egyptian Museum.

Use the provided museum documents as your PRIMARY source.
Enrich your answer with historical context, interesting facts, and
vivid descriptions to bring ancient Egypt to life for the visitor.
Never invent specific artifacts, dates, or measurements not in the sources.

Think step by step:
1. What historical topic is the visitor asking about?
2. What do the museum's own documents say?
3. What historical context makes this more fascinating?
4. Give a rich, educational, engaging answer.

MUSEUM SOURCES:
{context}

CONVERSATION HISTORY:
{chat_history}

QUESTION: {question}

HISTORICAL ANSWER:"""

    _VISITOR_EN = """\
You are a helpful visitor services guide at the Grand Egyptian Museum.

Use the provided sources to give accurate, practical visitor information.
Be specific about times, prices, and locations when available in sources.
If a specific detail (e.g. exact ticket price) is not in the sources, say:
"Please check gem.eg for the latest information on this."

Think step by step:
1. What practical information does the visitor need?
2. What do the sources say specifically?
3. Are there any helpful tips I can add?
4. Give a clear, practical, friendly answer.

MUSEUM SOURCES:
{context}

CONVERSATION HISTORY:
{chat_history}

VISITOR QUESTION: {question}

HELPFUL ANSWER:"""

    # ── Arabic Prompts ────────────────────────
    _GENERAL_AR = """\
أنت مرشد خبير للمتحف المصري الكبير.

استخدم مصادر المتحف المقدمة كمرجعك الأساسي.
يمكنك إثراء إجابتك بمعرفة علم المصريات ذات الصلة لتقديم
إجابة أكثر جاذبية واكتمالاً — لكن دائماً أعطِ الأولوية
لمحتوى المتحف نفسه ولا تخترع حقائق محددة.

إذا كان السؤال غامضاً أو غير واضح، اطرح سؤالاً توضيحياً ودياً.

فكر خطوة بخطوة:
1. ما الذي يسأل عنه الزائر حقاً؟
2. ماذا تقول المصادر؟
3. ما السياق الإضافي الذي يجعل الإجابة أغنى؟
4. قدم إجابة كاملة وجذابة وودية.

مصادر المتحف:
{context}

سجل المحادثة:
{chat_history}

سؤال الزائر: {question}

الإجابة:"""

    _HISTORY_AR = """\
أنت عالم آثار ومؤرخ متحمس في المتحف المصري الكبير.

استخدم وثائق المتحف المقدمة كمصدرك الأساسي.
أثرِ إجابتك بالسياق التاريخي والحقائق المثيرة للاهتمام
والأوصاف الحية لإحياء مصر القديمة للزائر.
لا تخترع قطعاً أثرية أو تواريخ أو قياسات محددة غير موجودة في المصادر.

فكر خطوة بخطوة:
1. ما الموضوع التاريخي الذي يسأل عنه الزائر؟
2. ماذا تقول وثائق المتحف؟
3. ما السياق التاريخي الذي يجعل هذا أكثر إثارة؟
4. قدم إجابة غنية وتعليمية وجذابة.

مصادر المتحف:
{context}

سجل المحادثة:
{chat_history}

السؤال: {question}

الإجابة التاريخية:"""

    _VISITOR_AR = """\
أنت مرشد خدمات الزوار في المتحف المصري الكبير.

استخدم المصادر المقدمة لتقديم معلومات زيارة دقيقة وعملية.
كن محدداً بشأن الأوقات والأسعار والمواقع عند توفرها في المصادر.
إذا لم تتوفر تفاصيل محددة (مثل سعر التذكرة بالضبط) في المصادر، قل:
"يرجى مراجعة gem.eg للحصول على أحدث المعلومات حول هذا."

فكر خطوة بخطوة:
1. ما المعلومات العملية التي يحتاجها الزائر؟
2. ماذا تقول المصادر تحديداً؟
3. هل هناك نصائح مفيدة يمكنني إضافتها؟
4. قدم إجابة واضحة وعملية وودية.

مصادر المتحف:
{context}

سجل المحادثة:
{chat_history}

سؤال الزائر: {question}

الإجابة المفيدة:"""

    _PROMPTS = {
        "en": {
            "general":      _GENERAL_EN,
            "history":      _HISTORY_EN,
            "visitor_info": _VISITOR_EN,
        },
        "ar": {
            "general":      _GENERAL_AR,
            "history":      _HISTORY_AR,
            "visitor_info": _VISITOR_AR,
        },
    }

    _NO_CONTEXT = {
        "en": (
            "I couldn't find specific information about your question "
            "in the museum's knowledge base.\n\n"
            "For accurate and up-to-date information, please:\n"
            "🌐 Visit the official website: gem.eg\n"
            "📧 Contact the museum directly\n\n"
            "Is there something else about the Grand Egyptian Museum "
            "I can help you with?"
        ),
        "ar": (
            "لم أتمكن من العثور على معلومات محددة حول سؤالك "
            "في قاعدة معرفة المتحف.\n\n"
            "للحصول على معلومات دقيقة ومحدّثة، يرجى:\n"
            "🌐 زيارة الموقع الرسمي: gem.eg\n"
            "📧 التواصل مع المتحف مباشرةً\n\n"
            "هل هناك شيء آخر يتعلق بالمتحف المصري الكبير "
            "يمكنني مساعدتك به؟"
        ),
    }

    def build_prompt(
        language:     str,
        query_type:   str,
        question:     str,
        context:      str,
        chat_history: str,
    ) -> dict:
        template = _PROMPTS[language].get(
            query_type, _PROMPTS[language]["general"]
        )
        return {
            "system": _SYSTEM[language],
            "user":   template.format(
                context=context,
                chat_history=chat_history or (
                    "لا توجد محادثة سابقة."
                    if language == "ar"
                    else "No previous conversation."
                ),
                question=question,
            ),
        }

    # ─────────────────────────────────────────
    # Load FAISS index + model
    # ─────────────────────────────────────────
    chunks_path = Path(VOLUME_PATH) / "chunks_metadata.json"
    index_path  = Path(VOLUME_PATH) / "faiss_index.bin"

    if not chunks_path.exists():
        raise FileNotFoundError(
            "chunks_metadata.json not found in volume."
        )
    if not index_path.exists():
        raise FileNotFoundError(
            "faiss_index.bin not found in volume."
        )

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    index = faiss.read_index(str(index_path))
    logger.info(
        f"✅ FAISS loaded: {index.ntotal} vectors | {len(chunks)} chunks"
    )

    # Load embedding model from volume cache
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    model_cache = str(Path(VOLUME_PATH) / "model_cache")
    Path(model_cache).mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        cache_folder=model_cache,
    )
    logger.info(f"✅ Embedding model loaded on {device}")

    # ─────────────────────────────────────────
    # Pipeline Execution
    # ─────────────────────────────────────────
    question   = request.question
    language   = detect_language(question)
    classifier = QueryClassifier()
    query_type = classifier.classify(question, language)

    logger.info(
        f"[{language.upper()}] [{query_type}] '{question[:80]}'"
    )

    # 1. Embed query
    query_vec = model.encode(
        e5_query(question),
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
    ).astype(np.float32).reshape(1, -1)

    # 2. FAISS search
    filter_active = bool(
        request.language_filter or request.category_filter
    )
    search_k = request.top_k * 10 if filter_active else request.top_k * 2

    scores, indices = index.search(query_vec, search_k)

    # 3. Dynamic threshold — gentler floor (0.60 multiplier)
    best_score    = float(scores[0][0]) if indices[0][0] != -1 else 0.0
    dynamic_floor = max(0.30, best_score * 0.60)

    logger.info(
        f"Best score: {best_score:.3f} | "
        f"Dynamic floor: {dynamic_floor:.3f}"
    )

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < dynamic_floor:
            continue
        chunk = chunks[idx].copy()
        chunk["similarity_score"] = float(score)
        if (request.language_filter
                and chunk.get("language") != request.language_filter):
            continue
        if (request.category_filter
                and chunk.get("category") != request.category_filter):
            continue
        results.append(chunk)
        if len(results) >= request.top_k:
            break

    logger.info(f"Retrieved {len(results)} chunks after filtering")

    # 4. No results
    if not results:
        return AskResponse(
            question=question,
            answer=_NO_CONTEXT[language],
            sources=[],
            query_type=query_type,
            query_language=language,
            chunks_retrieved=0,
        )

    # 5. Build context
    context_parts = []
    sources       = []
    for i, chunk in enumerate(results, 1):
        c_lang = chunk.get("language", "en")
        label  = f"[المصدر {i}]" if c_lang == "ar" else f"[Source {i}]"
        context_parts.append(
            f"{label}: {chunk['title']}\n{chunk['content']}"
        )
        sources.append(SourceModel(
            title    = chunk["title"],
            url      = chunk["url"],
            category = chunk["category"],
            language = c_lang,
            score    = round(chunk["similarity_score"], 3),
        ))

    context = truncate_context("\n\n---\n\n".join(context_parts))

    # 6. Build prompt
    prompt = build_prompt(
        language=language,
        query_type=query_type,
        question=question,
        context=context,
        chat_history=request.chat_history,
    )

    # 7. LLM call
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get(
            "OPENAI_BASE_URL",
            "https://api.openai.com/v1",
        ),
    )

    llm_model = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")

    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user",   "content": prompt["user"]},
            ],
            temperature=0.7,     # raised: more generative, fluid answers
            max_tokens=1200,     # raised: allows richer answers
        )
        answer = response.choices[0].message.content

    except Exception as e:
        logger.error(f"LLM error: {e}")
        answer = (
            "أعتذر، حدث خطأ أثناء توليد الإجابة. "
            "يرجى المحاولة مرة أخرى."
            if language == "ar"
            else "I apologize, an error occurred. Please try again."
        )

    return AskResponse(
        question=question,
        answer=answer,
        sources=sources,
        query_type=query_type,
        query_language=language,
        chunks_retrieved=len(results),
    )


# ─────────────────────────────────────────────
# Web Endpoint
# ─────────────────────────────────────────────
@app.function(
    image=pipeline_image,
    secrets=[modal.Secret.from_name("gem-secrets")],
    timeout=300,
    memory=8192,
)
@modal.fastapi_endpoint(method="POST", label="gem-ask")
def ask(request: AskRequest) -> AskResponse:
    """HTTP endpoint — receives Streamlit POST requests."""
    return process_rag.remote(request)


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────
@app.function(
    image=pipeline_image,
    volumes={VOLUME_PATH: volume},
    timeout=60,
    memory=512,
)
@modal.fastapi_endpoint(method="GET", label="gem-health")
def health() -> dict:
    """Quick health check — verifies volume files exist."""
    files = {
        "faiss_index":     (Path(VOLUME_PATH) / "faiss_index.bin").exists(),
        "chunks_metadata": (Path(VOLUME_PATH) / "chunks_metadata.json").exists(),
        "embeddings":      (Path(VOLUME_PATH) / "embeddings.npy").exists(),
        "model_cache":     (Path(VOLUME_PATH) / "model_cache").exists(),
    }
    return {
        "status": "healthy" if all(files.values()) else "degraded",
        "files":  files,
    }


# ─────────────────────────────────────────────
# Local Test Entrypoint
# ─────────────────────────────────────────────
@app.local_entrypoint()
def main():
    import time

    print("🚀 Testing GEM Pipeline in the cloud...")
    print("=" * 60)

    test_questions = [
        "What are the opening hours of the Grand Egyptian Museum?",
        "Tell me about Tutankhamun's golden mask",
        "ما هي مواعيد فتح المتحف المصري الكبير؟",
        "أخبرني عن قناع توت عنخ آمون الذهبي",
    ]

    for question in test_questions:
        result = process_rag.remote(AskRequest(question=question))

        print(f"\n❓ {question}")
        print(f"🌐 [{result.query_language.upper()}] [{result.query_type}]")
        print(f"💬 {result.answer[:400]}...")
        print(f"📚 Sources: {len(result.sources)}")
        for s in result.sources:
            print(
                f"   [{s.language.upper()}] "
                f"[{s.score:.3f}] "
                f"{s.title[:50]}"
            )
        print("-" * 60)
        time.sleep(15)   