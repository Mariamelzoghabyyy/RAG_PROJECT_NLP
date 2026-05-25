"""
gem_streamlit.py

Streamlit Chat Interface for GEM RAG Assistant.
Calls the Modal web endpoint via HTTP — no local ML dependencies needed.

Setup:
    pip install streamlit requests
    
    # Set your Modal endpoint URL (after: modal deploy gem_pipeline.py)
    # Either hardcode below or set env var:
    set MODAL_ASK_URL=https://yourname--gem-ask.modal.run
    set MODAL_HEALTH_URL=https://yourname--gem-health.modal.run

Run:
    streamlit run gem_streamlit.py
"""

import streamlit as st
import requests
import os
import time
from datetime import datetime
from typing import Optional

# ─────────────────────────────────────────────
# Modal Endpoint URLs
# ─────────────────────────────────────────────
# After running: modal deploy gem_pipeline.py
# Modal prints your URLs — paste them here
MODAL_ASK_URL    = os.getenv(
    "MODAL_ASK_URL",
    "https://tahermariam08--gem-ask.modal.run", # ← replace after deploy
)
MODAL_HEALTH_URL = os.getenv(
    "MODAL_HEALTH_URL",
    "https://tahermariam08--gem-health.modal.run", # ← replace after deploy
)

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title   = "GEM AI Assistant | المتحف المصري الكبير",
    page_icon    = "🏛️",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #8B6914, #D4A855);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
    }
    .source-card {
        background: #FFF9F0;
        border: 1px solid #D4A855;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.85em;
    }
    .badge {
        background: #8B6914;
        color: white;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.75em;
        margin: 2px;
        display: inline-block;
    }
    .badge-green {
        background: #2d7a2d;
        color: white;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.75em;
        margin: 2px;
        display: inline-block;
    }
    .status-ok   { color: #2d7a2d; font-weight: bold; }
    .status-fail { color: #cc0000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "messages":        [],
        "total_questions": 0,
        "chat_history":    [],   # list of {question, answer} for context
        "endpoint_ok":     None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()


# ─────────────────────────────────────────────
# Modal API Calls
# ─────────────────────────────────────────────
def check_health() -> dict:
    """Ping the Modal health endpoint."""
    try:
        r = requests.get(MODAL_HEALTH_URL, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


# In gem_streamlit.py — replace the call_ask function

def call_ask(
    question:        str,
    top_k:           int   = 5,
    score_threshold: float = 0.30,
    language_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
) -> dict:
    """POST to Modal endpoint with proper chat history."""

    # Build last 6 turns as formatted string
    history_lines = []
    turns = [
        m for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ]
    for msg in turns[-6:]:
        role = "Visitor" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:300]
        history_lines.append(f"{role}: {content}")
    chat_history = "\n".join(history_lines)

    payload = {
        "question":        question,
        "top_k":           top_k,
        "score_threshold": score_threshold,
        "language_filter": language_filter,
        "category_filter": category_filter,
        "chat_history":    chat_history,
    }

    try:
        r = requests.post(
            MODAL_ASK_URL,
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        return {
            "error":  "timeout",
            "answer": (
                "⏱️ Request timed out. "
                "Modal may be starting up — please try again."
            ),
        }
    except requests.exceptions.ConnectionError:
        return {
            "error":  "connection",
            "answer": (
                f"❌ Cannot reach Modal endpoint.\n"
                f"Check MODAL_ASK_URL: `{MODAL_ASK_URL}`"
            ),
        }
    except Exception as e:
        return {
            "error":  str(e),
            "answer": f"❌ Error: {str(e)}",
        }


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
def render_sidebar() -> tuple:
    with st.sidebar:

        st.markdown("## 🏛️ GEM AI Assistant")
        st.markdown("*Powered by Modal + GPT-4o-mini*")
        st.divider()

        # ── Endpoint status ───────────────────
        st.markdown("### 🔌 Modal Endpoint")

        if st.button("Check Connection", use_container_width=True):
            with st.spinner("Checking..."):
                health = check_health()
            st.session_state.endpoint_ok = (
                health.get("status") == "healthy"
            )

        if st.session_state.endpoint_ok is True:
            st.markdown(
                '<span class="status-ok">✅ Connected</span>',
                unsafe_allow_html=True,
            )
        elif st.session_state.endpoint_ok is False:
            st.markdown(
                '<span class="status-fail">❌ Disconnected</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("*Click 'Check Connection' to verify*")

        st.divider()

        # ── Search settings ───────────────────
        st.markdown("### ⚙️ Search Settings")

        top_k = st.slider(
            "Results to retrieve",
            min_value=1, max_value=10, value=5,
        )

        score_threshold = st.slider(
            "Minimum similarity score",
            min_value=0.0, max_value=1.0, value=0.45, step=0.05,
        )

        category_filter = st.selectbox(
            "Filter by category",
            options=[
                "All", "collection", "visitor_info",
                "educational", "events", "about", "exhibition",
            ],
            index=0,
        )

        language_filter = st.selectbox(
            "Filter by language",
            options=["Both", "English only (en)", "Arabic only (ar)"],
            index=0,
        )

        st.divider()

        # ── Quick questions ───────────────────
        st.markdown("### 💡 Quick Questions")

        quick_en = [
            "What are the opening hours?",
            "How do I buy tickets?",
            "Tell me about Tutankhamun's mask",
            "Are there educational programs?",
            "Is there parking available?",
        ]
        quick_ar = [
            "ما هي مواعيد فتح المتحف؟",
            "كيف أشتري التذاكر؟",
            "أخبرني عن قناع توت عنخ آمون",
            "هل توجد برامج تعليمية؟",
        ]

        st.markdown("**English:**")
        for q in quick_en:
            if st.button(q, key=f"en_{q}", use_container_width=True):
                st.session_state["pending_question"] = q

        st.markdown("**عربي:**")
        for q in quick_ar:
            if st.button(q, key=f"ar_{q}", use_container_width=True):
                st.session_state["pending_question"] = q

        st.divider()

        # ── Stats ─────────────────────────────
        st.markdown("### 📊 Session")
        st.metric("Questions Asked", st.session_state.total_questions)

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages        = []
            st.session_state.total_questions = 0
            st.session_state.chat_history    = []
            st.rerun()

        # Parse filter values
        lang_map = {
            "Both":              None,
            "English only (en)": "en",
            "Arabic only (ar)":  "ar",
        }

        return (
            top_k,
            score_threshold,
            None if category_filter == "All" else category_filter,
            lang_map[language_filter],
        )


# ─────────────────────────────────────────────
# Chat Messages Renderer
# ─────────────────────────────────────────────
def render_messages():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧑‍💼"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🏛️"):
                st.write(msg["content"])

                if msg.get("sources"):
                    with st.expander(
                        f"📚 {len(msg['sources'])} source(s) used",
                        expanded=False,
                    ):
                        for src in msg["sources"]:
                            lang_flag = "🇪🇬" if src["language"] == "ar" else "🇬🇧"
                            st.markdown(f"""
                            <div class="source-card">
                                {lang_flag} <strong>{src['title']}</strong><br>
                                🔗 <a href="{src['url']}" target="_blank">
                                    {src['url'][:70]}
                                </a><br>
                                <span class="badge">{src['category']}</span>
                                <span class="badge">score: {src['score']:.2f}</span>
                                <span class="badge">{src['language'].upper()}</span>
                            </div>
                            """, unsafe_allow_html=True)

                if msg.get("meta"):
                    meta = msg["meta"]
                    cols = st.columns(3)
                    cols[0].caption(f"🔍 {meta.get('query_type','?')}")
                    cols[1].caption(f"📦 {meta.get('chunks_retrieved','?')} chunks")
                    cols[2].caption(f"🌐 {meta.get('query_language','?').upper()}")


# ─────────────────────────────────────────────
# Process Question
# ─────────────────────────────────────────────
def process_question(
    question:        str,
    top_k:           int,
    score_threshold: float,
    category_filter: Optional[str],
    language_filter: Optional[str],
):
    """Send question to Modal, display response, save to state."""
    # Add user message
    st.session_state.messages.append({
        "role":    "user",
        "content": question,
    })

    with st.chat_message("user", avatar="🧑‍💼"):
        st.write(question)

    with st.chat_message("assistant", avatar="🏛️"):
        with st.spinner("🔍 Searching museum knowledge base..."):
            start = time.time()
            result = call_ask(
                question=question,
                top_k=top_k,
                score_threshold=score_threshold,
                language_filter=language_filter,
                category_filter=category_filter,
            )
            elapsed = round((time.time() - start) * 1000)

        # Display answer
        if "error" in result:
            st.error(result["answer"])
            return

        answer  = result.get("answer", "No answer returned.")
        sources = result.get("sources", [])
        meta    = {
            "query_type":       result.get("query_type", "?"),
            "query_language":   result.get("query_language", "?"),
            "chunks_retrieved": result.get("chunks_retrieved", 0),
            "response_ms":      elapsed,
        }

        st.write(answer)

        # Sources expander
        if sources:
            with st.expander(f"📚 {len(sources)} source(s) used", expanded=False):
                for src in sources:
                    lang_flag = "🇪🇬" if src["language"] == "ar" else "🇬🇧"
                    st.markdown(f"""
                    <div class="source-card">
                        {lang_flag} <strong>{src['title']}</strong><br>
                        🔗 <a href="{src['url']}" target="_blank">
                            {src['url'][:70]}
                        </a><br>
                        <span class="badge">{src['category']}</span>
                        <span class="badge">score: {src['score']:.2f}</span>
                        <span class="badge">{src['language'].upper()}</span>
                    </div>
                    """, unsafe_allow_html=True)

        # Meta row
        cols = st.columns(4)
        cols[0].caption(f"🔍 {meta['query_type']}")
        cols[1].caption(f"📦 {meta['chunks_retrieved']} chunks")
        cols[2].caption(f"🌐 {meta['query_language'].upper()}")
        cols[3].caption(f"⚡ {meta['response_ms']}ms")

    # Save to session
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
        "meta":    meta,
    })
    st.session_state.chat_history.append({
        "question": question,
        "answer":   answer,
    })
    if len(st.session_state.chat_history) > 10:
        st.session_state.chat_history = st.session_state.chat_history[-10:]

    st.session_state.total_questions += 1


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ Grand Egyptian Museum</h1>
        <h3>AI Assistant | المساعد الذكي</h3>
        <p>Ask me anything in English or Arabic • اسألني بالعربية أو الإنجليزية</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    top_k, score_threshold, category_filter, language_filter = render_sidebar()

    # URL config notice
    if "YOUR_WORKSPACE" in MODAL_ASK_URL:
        st.warning(
            "⚠️ **Setup required:** Replace `YOUR_WORKSPACE` in `MODAL_ASK_URL` "
            "with your actual Modal endpoint URL.\n\n"
            "Run `modal deploy gem_pipeline.py` to get your URL.",
            icon="⚠️",
        )

    # Welcome
    if not st.session_state.messages:
        col1, col2, col3, col4 = st.columns(4)
        col1.info("🏺 **Collections**\nArtifacts & Exhibits")
        col2.info("🎫 **Tickets**\nPrices & Booking")
        col3.info("📅 **Events**\nPrograms & Tours")
        col4.info("📚 **Education**\nSchools & Groups")
        st.markdown(
            "### 👋 Welcome! Type your question below or pick one from the sidebar."
        )

    # Render history
    render_messages()

    # Handle quick question buttons
    pending = st.session_state.pop("pending_question", None)

    # Chat input
    chat_input = st.chat_input(
        "Ask about the Grand Egyptian Museum... | اسأل عن المتحف المصري الكبير...",
    )

    question = pending or chat_input

    if question:
        process_question(
            question=question,
            top_k=top_k,
            score_threshold=score_threshold,
            category_filter=category_filter,
            language_filter=language_filter,
        )


if __name__ == "__main__":
    main()