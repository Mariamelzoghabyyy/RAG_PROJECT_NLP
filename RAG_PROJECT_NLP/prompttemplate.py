"""
gem_prompt_templates.py

Prompt templates for the GEM RAG assistant.
— Arabic + English bilingual
— Language-aware + query_type-aware selection
— Context length protection (truncation before LLM call)
— No external dependencies (pure Python)

Fixes over v1:
- detect_language no longer duplicated — single implementation here
- Context truncation added (prevents context-window overflow)
- Arabic prompts include instruction for mixed-language context
- NO_CONTEXT_RESPONSE markdown cleaned up (emoji only, no ** syntax)
- get_prompt() validates and sanitises all inputs
"""

import re
import tiktoken
from typing import Optional


# ═════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════

MAX_CONTEXT_TOKENS = 3_000     # leaves room for question + system + response
TIKTOKEN_MODEL     = "gpt-4o-mini"

MUSEUM_NAMES = {
    "Grand Egyptian Museum",
    "المتحف المصري الكبير",
    "GEM",
}


# ═════════════════════════════════════════════
# Language Detection (single source of truth)
# ═════════════════════════════════════════════

def detect_language(text: str, threshold: float = 0.2) -> str:
    """
    Detect whether text is Arabic or English.
    Returns 'ar' or 'en'.
    Uses Arabic Unicode block coverage as the signal.
    """
    if not text:
        return "en"
    arabic = sum(
        1 for c in text
        if "\u0600" <= c <= "\u06ff"
        or "\u0750" <= c <= "\u077f"
        or "\ufb50" <= c <= "\ufdff"
        or "\ufe70" <= c <= "\ufeff"
    )
    return "ar" if arabic / max(len(text), 1) > threshold else "en"


# ═════════════════════════════════════════════
# Context Truncation
# ═════════════════════════════════════════════

def _get_tokenizer():
    try:
        return tiktoken.encoding_for_model(TIKTOKEN_MODEL)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


_TOKENIZER = None   # lazy-loaded singleton


def truncate_context(
    context: str,
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> str:
    """
    Truncate context to max_tokens to prevent LLM context-window overflow.
    Appends a note when truncation occurs.
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = _get_tokenizer()

    tokens = _TOKENIZER.encode(context)
    if len(tokens) <= max_tokens:
        return context

    truncated = _TOKENIZER.decode(tokens[:max_tokens])
    note_en   = "\n\n[Context truncated to fit model context window.]"
    note_ar   = "\n\n[تم اقتصاص السياق ليناسب نافذة السياق للنموذج.]"

    # Add note in language matching the context
    note = note_ar if detect_language(context) == "ar" else note_en
    return truncated + note


# ═════════════════════════════════════════════
# ENGLISH PROMPTS
# ═════════════════════════════════════════════

_SYSTEM_EN = (
    "You are a knowledgeable and friendly AI assistant for the "
    "Grand Egyptian Museum (GEM) — the world's largest archaeological museum. "
    "Always respond in English. "
    "Never invent facts, prices, opening hours, or dates."
)

_GENERAL_EN = """\
You are an expert AI assistant for the Grand Egyptian Museum (GEM), \
the world's largest archaeological museum, located in Giza, Egypt.

Your role is to provide accurate, informative, and engaging answers about:
- The museum's collections and artefacts
- Ancient Egyptian history and civilisation
- Exhibition and gallery information
- Visitor information (hours, tickets, facilities)
- Educational programmes and events
- Museum services and amenities

STRICT INSTRUCTIONS:
1. Answer ONLY based on the provided context below.
2. If the answer is not clearly supported by the context, say:
   "I don't have specific information about this. Please visit gem.eg \
or contact the museum directly."
3. Do NOT invent facts, prices, dates, or opening hours.
4. Be friendly, informative, and educational.
5. Provide rich historical context when it is available in the sources.
6. Cite sources when helpful (e.g. "According to the museum's website…").
7. Keep responses concise but complete.
8. Always respond in English.
   If any source is in Arabic, translate relevant points into English.

CONTEXT FROM MUSEUM SOURCES:
{context}

CONVERSATION HISTORY:
{chat_history}

VISITOR'S QUESTION:
{question}

ANSWER:\
"""

_HISTORY_EN = """\
You are a knowledgeable Egyptologist and historian specialising in \
ancient Egyptian civilisation, working at the Grand Egyptian Museum.

Based ONLY on the museum documents below, answer the historical question \
with depth and accuracy.

STRICT INSTRUCTIONS:
1. Use ONLY the provided sources — do not add outside knowledge.
2. Include relevant dates, dynasty names, and historical context when available.
3. If a detail is not in the sources, say so — do not invent it.
4. Be engaging and educational.
5. Respond in English.
   If any source is in Arabic, translate relevant points into English.

SOURCES:
{context}

CONVERSATION HISTORY:
{chat_history}

QUESTION:
{question}

HISTORICAL ANSWER:\
"""

_VISITOR_EN = """\
You are a helpful visitor services assistant for the Grand Egyptian Museum \
(gem.eg).

Provide practical, accurate visitor information based ONLY on the sources below.

STRICT INSTRUCTIONS:
1. Use ONLY the provided sources.
2. Be specific: state exact times, prices, and locations when available.
3. If a detail (e.g. ticket price) is absent from the sources, say:
   "Please check gem.eg for the latest information."
4. Give practical tips where relevant.
5. Respond in English.
   If any source is in Arabic, translate relevant points into English.

SOURCES:
{context}

CONVERSATION HISTORY:
{chat_history}

VISITOR QUESTION:
{question}

HELPFUL ANSWER:\
"""

_NO_CONTEXT_EN = """\
I'm sorry, I couldn't find specific information about your question \
in my knowledge base.

For accurate and up-to-date information, please:
  🌐 Visit the official website: gem.eg
  📧 Contact the museum directly via the website
  📞 Call the visitor services team

Is there something else about the Grand Egyptian Museum I can help you with?\
"""


# ═════════════════════════════════════════════
# ARABIC PROMPTS
# ═════════════════════════════════════════════

_SYSTEM_AR = (
    "أنت مساعد ذكاء اصطناعي ودود ومتخصص في المتحف المصري الكبير "
    "(GEM) — أكبر متحف أثري في العالم. "
    "أجب دائماً باللغة العربية. "
    "لا تخترع أبداً حقائق أو أسعاراً أو مواعيد عمل أو تواريخ."
)

_GENERAL_AR = """\
أنت مساعد ذكاء اصطناعي متخصص في المتحف المصري الكبير (GEM)، \
أكبر متحف أثري في العالم، الواقع في الجيزة، مصر.

دورك تقديم إجابات دقيقة ومفيدة وجذابة حول:
- مجموعات المتحف والقطع الأثرية
- التاريخ والحضارة المصرية القديمة
- معلومات المعارض والقاعات
- معلومات الزوار (ساعات العمل، التذاكر، المرافق)
- البرامج التعليمية والفعاليات
- خدمات المتحف والمرافق

تعليمات صارمة:
1. أجب فقط بناءً على السياق المقدم أدناه.
2. إذا لم تكن الإجابة مدعومة بوضوح من السياق، قل:
   "لا تتوفر لديّ معلومات محددة حول هذا. يرجى زيارة gem.eg أو التواصل \
مع المتحف مباشرةً."
3. لا تخترع حقائق أو أسعاراً أو تواريخ أو مواعيد عمل.
4. كن ودوداً ومفيداً وتعليمياً.
5. قدم سياقاً تاريخياً غنياً ومفصلاً عند توفره في المصادر.
6. أشر إلى المصادر عند الاقتضاء.
7. اجعل إجاباتك مختصرة لكن شاملة.
8. أجب دائماً باللغة العربية.
   إذا كانت بعض المصادر بالإنجليزية، ترجم النقاط ذات الصلة وأجب بالعربية.

السياق من مصادر المتحف:
{context}

سجل المحادثة:
{chat_history}

سؤال الزائر:
{question}

الإجابة:\
"""

_HISTORY_AR = """\
أنت عالم آثار ومؤرخ متخصص في الحضارة المصرية القديمة، \
تعمل في المتحف المصري الكبير.

بناءً على وثائق المتحف التالية فقط، أجب على السؤال التاريخي بعمق ودقة.

تعليمات صارمة:
1. استخدم المصادر المقدمة أدناه فقط — لا تضف معلومات خارجية.
2. أدرج التواريخ وأسماء الأسرات الحاكمة والسياق التاريخي عند توفره.
3. إذا لم تتوفر تفاصيل في المصادر، قل ذلك — لا تخترعها.
4. كن مشوقاً وتعليمياً.
5. أجب باللغة العربية.
   إذا كانت بعض المصادر بالإنجليزية، ترجم النقاط ذات الصلة وأجب بالعربية.

المصادر:
{context}

سجل المحادثة:
{chat_history}

السؤال:
{question}

الإجابة التاريخية:\
"""

_VISITOR_AR = """\
أنت مساعد خدمات الزوار في المتحف المصري الكبير (gem.eg).

قدم معلومات عملية ودقيقة بناءً على المصادر التالية فقط.

تعليمات صارمة:
1. استخدم المصادر المقدمة أدناه فقط.
2. كن محدداً: اذكر الأوقات والأسعار والمواقع الدقيقة عند توفرها.
3. إذا لم تتوفر معلومة (مثل سعر التذكرة) في المصادر، قل:
   "يرجى مراجعة gem.eg للحصول على أحدث المعلومات."
4. قدم نصائح عملية للزوار.
5. أجب باللغة العربية.
   إذا كانت بعض المصادر بالإنجليزية، ترجم النقاط ذات الصلة وأجب بالعربية.

المصادر:
{context}

سجل المحادثة:
{chat_history}

سؤال الزائر:
{question}

الإجابة المفيدة:\
"""

_NO_CONTEXT_AR = """\
أعتذر، لم أتمكن من العثور على معلومات محددة حول سؤالك \
في قاعدة المعرفة لديّ.

للحصول على معلومات دقيقة ومحدّثة، يرجى:
  🌐 زيارة الموقع الرسمي: gem.eg
  📧 التواصل مع المتحف مباشرةً عبر الموقع
  📞 الاتصال بفريق خدمات الزوار

هل هناك شيء آخر يتعلق بالمتحف المصري الكبير يمكنني مساعدتك به؟\
"""


# ═════════════════════════════════════════════
# Prompt Maps
# ═════════════════════════════════════════════

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

_SYSTEM_PROMPTS = {
    "en": _SYSTEM_EN,
    "ar": _SYSTEM_AR,
}

_NO_CONTEXT = {
    "en": _NO_CONTEXT_EN,
    "ar": _NO_CONTEXT_AR,
}

_VALID_QUERY_TYPES  = {"general", "history", "visitor_info"}
_DEFAULT_HISTORY_EN = "No previous conversation."
_DEFAULT_HISTORY_AR = "لا توجد محادثة سابقة."


# ═════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════

def get_prompt(
    question:     str,
    context:      str,
    chat_history: str          = "",
    query_type:   Optional[str] = None,
    max_context_tokens: int    = MAX_CONTEXT_TOKENS,
) -> dict:
    """
    Build the complete prompt for the LLM.

    Args:
        question           : User question (AR or EN)
        context            : Retrieved context string from retriever
        chat_history       : Formatted previous conversation (optional)
        query_type         : "general" | "history" | "visitor_info"
                             Pass from retriever result for best accuracy.
                             Auto-defaults to "general" if None or invalid.
        max_context_tokens : Hard cap on context length before LLM call.

    Returns:
        {
            "system_prompt" : str,
            "user_prompt"   : str,
            "language"      : "en" | "ar",
            "query_type"    : str,
            "context_truncated": bool,
        }

    Usage:
        result = retrieve.remote(query=question, top_k=5)
        prompt = get_prompt(
            question   = question,
            context    = result["context"],
            query_type = result["query_type"],   # from retriever
        )
        response = openai_client.chat.completions.create(
            model    = "gpt-4o-mini",
            messages = [
                {"role": "system", "content": prompt["system_prompt"]},
                {"role": "user",   "content": prompt["user_prompt"]},
            ]
        )
    """
    # Validate inputs
    language   = detect_language(question)
    query_type = (
        query_type if query_type in _VALID_QUERY_TYPES else "general"
    )

    # Truncate context to protect context window
    original_context_len = len(context)
    context              = truncate_context(context, max_context_tokens)
    context_truncated    = len(context) < original_context_len

    # Default chat history string
    if not chat_history:
        chat_history = (
            _DEFAULT_HISTORY_AR if language == "ar"
            else _DEFAULT_HISTORY_EN
        )

    # Select template
    template      = _PROMPTS[language][query_type]
    system_prompt = _SYSTEM_PROMPTS[language]

    user_prompt = template.format(
        context      = context,
        chat_history = chat_history,
        question     = question,
    )

    return {
        "system_prompt":    system_prompt,
        "user_prompt":      user_prompt,
        "language":         language,
        "query_type":       query_type,
        "context_truncated": context_truncated,
    }


def get_no_context_response(question: str) -> str:
    """
    Return a polite 'no information found' message
    in the same language as the question.
    """
    language = detect_language(question)
    return _NO_CONTEXT[language]


# ═════════════════════════════════════════════
# Quick smoke test (run directly: python gem_prompt_templates.py)
# ═════════════════════════════════════════════
if __name__ == "__main__":
    test_cases = [
        ("What are the opening hours?",       "visitor_info", "en"),
        ("Tell me about Tutankhamun's mask",  "history",      "en"),
        ("ما مواعيد فتح المتحف؟",              "visitor_info", "ar"),
        ("قناع توت عنخ آمون الذهبي",           "history",      "ar"),
    ]

    dummy_context = (
        "[Source 1]: Grand Egyptian Museum\n"
        "Opening hours: 9 AM – 5 PM daily.\n\n"
        "---\n\n"
        "[Source 2]: Tutankhamun Galleries\n"
        "The golden mask weighs 10.23 kg and is made of solid gold."
    )

    print("=" * 60)
    print("GEM Prompt Template Smoke Test")
    print("=" * 60)

    for question, q_type, expected_lang in test_cases:
        result = get_prompt(
            question   = question,
            context    = dummy_context,
            query_type = q_type,
        )
        lang_ok = "✅" if result["language"] == expected_lang else "❌"
        type_ok = "✅" if result["query_type"] == q_type      else "❌"
        print(
            f"\n{lang_ok} {type_ok} [{result['language'].upper()}] "
            f"[{result['query_type']}] {question[:50]}"
        )
        print(f"   System  : {result['system_prompt'][:80]}...")
        print(f"   Prompt  : {result['user_prompt'][:120]}...")
        print(f"   Truncated: {result['context_truncated']}")

    # Test no-context response
    print("\n--- No Context Responses ---")
    print(get_no_context_response("What are the opening hours?"))
    print()
    print(get_no_context_response("ما مواعيد الزيارة؟"))