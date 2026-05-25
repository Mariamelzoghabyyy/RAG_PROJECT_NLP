"""
gem_evaluator.py

Evaluates the GEM RAG system by calling the Modal endpoint.
Runs locally — no GPU, no Modal function needed.

Run:
    pip install requests
    python gem_evaluator.py
"""

import json
import time
import logging
import requests
import os
from typing import List, Dict
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODAL_ASK_URL = os.getenv(
    "MODAL_ASK_URL",
    "https://YOUR_WORKSPACE--gem-ask.modal.run",
)

# ─────────────────────────────────────────────
# Test Dataset — English + Arabic
# ─────────────────────────────────────────────
TEST_QUESTIONS = [
    # ── English ──────────────────────────────
    {
        "id":               "EN-001",
        "question":         "What are the opening hours of the Grand Egyptian Museum?",
        "expected_keywords": ["hours", "open", "visit", "time"],
        "category":         "visitor_info",
        "language":         "en",
        "difficulty":       "easy",
    },
    {
        "id":               "EN-002",
        "question":         "Tell me about Tutankhamun's golden mask",
        "expected_keywords": ["tutankhamun", "gold", "mask", "pharaoh"],
        "category":         "collection",
        "language":         "en",
        "difficulty":       "medium",
    },
    {
        "id":               "EN-003",
        "question":         "How much do tickets to the GEM cost?",
        "expected_keywords": ["ticket", "price", "cost", "admission"],
        "category":         "visitor_info",
        "language":         "en",
        "difficulty":       "easy",
    },
    {
        "id":               "EN-004",
        "question":         "What is the significance of Ramesses II?",
        "expected_keywords": ["ramesses", "pharaoh", "dynasty", "egypt"],
        "category":         "collection",
        "language":         "en",
        "difficulty":       "hard",
    },
    {
        "id":               "EN-005",
        "question":         "Are there educational programs for school groups?",
        "expected_keywords": ["school", "education", "program", "children"],
        "category":         "educational",
        "language":         "en",
        "difficulty":       "medium",
    },
    # ── Arabic ────────────────────────────────
    {
        "id":               "AR-001",
        "question":         "ما هي مواعيد فتح المتحف المصري الكبير؟",
        "expected_keywords": ["مواعيد", "فتح", "ساعات", "زيارة"],
        "category":         "visitor_info",
        "language":         "ar",
        "difficulty":       "easy",
    },
    {
        "id":               "AR-002",
        "question":         "أخبرني عن قناع توت عنخ آمون الذهبي",
        "expected_keywords": ["توت", "قناع", "ذهب", "فرعون"],
        "category":         "collection",
        "language":         "ar",
        "difficulty":       "medium",
    },
    {
        "id":               "AR-003",
        "question":         "كم سعر تذكرة الدخول للمتحف؟",
        "expected_keywords": ["تذكرة", "سعر", "دخول"],
        "category":         "visitor_info",
        "language":         "ar",
        "difficulty":       "easy",
    },
    {
        "id":               "AR-004",
        "question":         "ما أهمية رمسيس الثاني في التاريخ المصري؟",
        "expected_keywords": ["رمسيس", "فرعون", "أسرة", "تاريخ"],
        "category":         "collection",
        "language":         "ar",
        "difficulty":       "hard",
    },
    {
        "id":               "AR-005",
        "question":         "هل توجد برامج تعليمية للمدارس في المتحف؟",
        "expected_keywords": ["تعليم", "مدارس", "برامج", "أطفال"],
        "category":         "educational",
        "language":         "ar",
        "difficulty":       "medium",
    },
]


# ─────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────
class RAGEvaluator:
    """
    Calls the Modal endpoint and evaluates:
    1. Retrieval quality (keyword recall in sources)
    2. Answer relevance (expected keywords in answer)
    3. Language correctness (response language matches question language)
    4. Response time
    5. No-hallucination proxy (answer not empty / error)
    """

    def __init__(self, modal_url: str):
        self.modal_url = modal_url
        self.results: List[Dict] = []

    def call_endpoint(self, question: str) -> dict:
        """Call Modal /ask endpoint."""
        try:
            r = requests.post(
                self.modal_url,
                json={"question": question, "top_k": 5},
                timeout=120,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Endpoint error: {e}")
            return {"error": str(e), "answer": "", "sources": []}

    def evaluate_one(self, test: Dict) -> Dict:
        question = test["question"]
        expected = test["expected_keywords"]

        logger.info(
            f"Testing {test['id']} [{test['language'].upper()}]: "
            f"{question[:60]}..."
        )

        start  = time.time()
        result = self.call_endpoint(question)
        elapsed_ms = round((time.time() - start) * 1000, 1)

        if "error" in result:
            return {
                "test_id":    test["id"],
                "question":   question,
                "language":   test["language"],
                "difficulty": test["difficulty"],
                "error":      result["error"],
                "overall_score": 0.0,
                "response_time_ms": elapsed_ms,
            }

        answer        = result.get("answer", "")
        sources       = result.get("sources", [])
        response_lang = result.get("query_language", "?")
        chunks        = result.get("chunks_retrieved", 0)

        # ── Metric 1: Keyword recall in answer ────
        answer_lower   = answer.lower()
        kws_in_answer  = sum(
            1 for kw in expected
            if kw.lower() in answer_lower
        )
        keyword_coverage = kws_in_answer / max(len(expected), 1)

        # ── Metric 2: Source keyword recall ───────
        sources_text = " ".join(
            (s.get("title", "") + " " + s.get("url", "")).lower()
            for s in sources
        )
        kws_in_sources = sum(
            1 for kw in expected
            if kw.lower() in sources_text
        )
        source_recall = kws_in_sources / max(len(expected), 1)

        # ── Metric 3: Language correctness ────────
        lang_correct = (response_lang == test["language"])

        # ── Metric 4: Answer completeness ─────────
        word_count   = len(answer.split())
        is_complete  = word_count >= 30

        # ── Metric 5: Not a no-info response ──────
        no_info_phrases = [
            "i don't have", "i couldn't find", "no information",
            "لا تتوفر", "لم أتمكن", "لا توجد معلومات",
        ]
        has_info = not any(p in answer.lower() for p in no_info_phrases)

        # ── Overall score ──────────────────────────
        overall = (
            keyword_coverage * 0.35 +
            source_recall    * 0.30 +
            (1.0 if lang_correct else 0.0) * 0.20 +
            (1.0 if is_complete  else 0.5) * 0.10 +
            (1.0 if has_info     else 0.0) * 0.05
        )

        return {
            "test_id":          test["id"],
            "question":         question,
            "language":         test["language"],
            "difficulty":       test["difficulty"],
            "category":         test["category"],
            "answer_preview":   answer[:200],
            "chunks_retrieved": chunks,
            "sources_found":    len(sources),
            "keyword_coverage": round(keyword_coverage, 3),
            "source_recall":    round(source_recall, 3),
            "lang_correct":     lang_correct,
            "response_lang":    response_lang,
            "is_complete":      is_complete,
            "word_count":       word_count,
            "has_info":         has_info,
            "overall_score":    round(overall, 3),
            "response_time_ms": elapsed_ms,
        }

    def run(self, output_path: str = "gem_eval_results.json") -> Dict:
        print("\n🧪 GEM RAG Evaluation")
        print(f"   Endpoint  : {self.modal_url}")
        print(f"   Questions : {len(TEST_QUESTIONS)}")
        print("=" * 60)

        self.results = []
        for test in TEST_QUESTIONS:
            result = self.evaluate_one(test)
            self.results.append(result)

            score = result.get("overall_score", 0.0)
            flag  = "✅" if score >= 0.6 else "⚠️"
            print(
                f"{flag} {result['test_id']} [{result['language'].upper()}] "
                f"Score: {score:.2f} | "
                f"Coverage: {result.get('keyword_coverage', 0):.2f} | "
                f"{result.get('response_time_ms', 0):.0f}ms"
            )

        summary = self._summary()
        output  = {
            "evaluation_date": datetime.now().isoformat(),
            "endpoint":        self.modal_url,
            "total_questions": len(TEST_QUESTIONS),
            "summary":         summary,
            "results":         self.results,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        self._print_summary(summary)
        print(f"\n💾 Results saved → {output_path}")
        return summary

    def _summary(self) -> Dict:
        n  = len(self.results)
        ok = [r for r in self.results if "error" not in r]
        if not ok:
            return {"error": "All requests failed"}

        en = [r for r in ok if r.get("language") == "en"]
        ar = [r for r in ok if r.get("language") == "ar"]

        def avg(lst, key):
            vals = [r[key] for r in lst if key in r]
            return round(sum(vals) / max(len(vals), 1), 3)

        return {
            "total_questions":       n,
            "successful_calls":      len(ok),
            "failed_calls":          n - len(ok),
            "avg_overall_score":     avg(ok, "overall_score"),
            "avg_keyword_coverage":  avg(ok, "keyword_coverage"),
            "avg_source_recall":     avg(ok, "source_recall"),
            "avg_response_time_ms":  avg(ok, "response_time_ms"),
            "language_correct_pct":  round(
                sum(1 for r in ok if r.get("lang_correct")) / max(len(ok), 1), 3
            ),
            "complete_answers":      sum(1 for r in ok if r.get("is_complete")),
            "has_info_responses":    sum(1 for r in ok if r.get("has_info")),
            "en_avg_score":          avg(en, "overall_score"),
            "ar_avg_score":          avg(ar, "overall_score"),
        }

    def _print_summary(self, s: Dict):
        score = s.get("avg_overall_score", 0)
        grade = (
            "A 🌟 Excellent"    if score >= 0.80 else
            "B ✅ Good"         if score >= 0.70 else
            "C ⚠️  Fair"        if score >= 0.60 else
            "D ❌ Needs Work"
        )

        print(f"\n{'='*60}")
        print(f"📊 EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"  Questions          : {s['total_questions']}")
        print(f"  Successful calls   : {s['successful_calls']}")
        print(f"  Failed calls       : {s['failed_calls']}")
        print(f"  Avg overall score  : {s['avg_overall_score']:.2f} / 1.00")
        print(f"  Keyword coverage   : {s['avg_keyword_coverage']:.2f}")
        print(f"  Source recall      : {s['avg_source_recall']:.2f}")
        print(f"  Language correct   : {s['language_correct_pct']*100:.0f}%")
        print(f"  Complete answers   : {s['complete_answers']}/{s['total_questions']}")
        print(f"  Has info           : {s['has_info_responses']}/{s['total_questions']}")
        print(f"  Avg response time  : {s['avg_response_time_ms']:.0f}ms")
        print(f"  EN avg score       : {s['en_avg_score']:.2f}")
        print(f"  AR avg score       : {s['ar_avg_score']:.2f}")
        print(f"\n🏆 Grade: {grade}")
        print(f"{'='*60}")


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    evaluator = RAGEvaluator(modal_url=MODAL_ASK_URL)
    evaluator.run(output_path="gem_eval_results.json")