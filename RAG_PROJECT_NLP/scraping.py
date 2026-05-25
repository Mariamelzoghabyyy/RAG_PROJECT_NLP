"""
gem_eg_playwright_scraper.py

Modal-compatible Playwright scraper for gem.eg
Arabic + English bilingual
Saves to Modal Volume "rag"

Improvements over v1:
- Shared utils (detect_language, is_arabic_text, categorize_url)
- Fresh page per URL (crash isolation)
- Retry logic with exponential backoff
- Incremental checkpointing every 10 URLs
- CSS selector noise removal (faster than lambda)
- dict.fromkeys() deduplication
- networkidle wait for JS-heavy pages
- Cleaner error taxonomy
"""

import modal
from pathlib import Path

# ─────────────────────────────────────────────
# Modal App & Infrastructure
# ─────────────────────────────────────────────
app = modal.App("gem-scraper")

scraper_image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands(
        "apt-get update && apt-get install -y "
        "libnss3 libatk1.0-0 libatk-bridge2.0-0 "
        "libcups2 libdrm2 libxkbcommon0 libxcomposite1 "
        "libxdamage1 libxfixes3 libxrandr2 libgbm1 "
        "libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 "
        "libwayland-client0 wget ca-certificates fonts-liberation "
        "libappindicator3-1 xdg-utils "
        "fonts-arabeyes fonts-arphic-uming "
        "--no-install-recommends"
    )
    .pip_install([
        "playwright",
        "beautifulsoup4",
    ])
    .run_commands("playwright install chromium")
)

volume    = modal.Volume.from_name("rag", create_if_missing=True)
VOLUME_PATH = "/data"

# ─────────────────────────────────────────────
# Seed URLs
# ─────────────────────────────────────────────
SEED_URLS = [
    # ── English ──────────────────────────────
    "https://gem.eg/en",
    "https://gem.eg/en/visit",
    "https://gem.eg/en/about",
    "https://gem.eg/en/collection",
    "https://gem.eg/en/visit/plan-your-visit/",
    "https://gem.eg/en/visit/plan-your-visit/opening-hours/",
    "https://gem.eg/en/visit/plan-your-visit/visitor-tips/",
    "https://gem.eg/en/visit/plan-your-visit/museum-maps/",
    "https://gem.eg/en/visit/accessibility/",
    "https://gem.eg/en/gem-experience/arts-and-crafts-centre/",
    "https://gem.eg/en/gem-experience/arts-and-crafts-centre/about-arts-and-crafts/",
    "https://gem.eg/en/gem-experience/education-centre/",
    "https://gem.eg/en/gem-experience/education-centre/about-the-centre/",
    "https://gem.eg/en/gem-experience/children-museum/",
    "https://gem.eg/en/gem-experience/children-museum/about-the-museum/",
    "https://gem.eg/en/gem-experience/children-museum/cm-galleries/",
    "https://gem.eg/en/collection/grand-hall/",
    "https://gem.eg/en/collection/grand-stairs/",
    "https://gem.eg/en/collection/tutankhamun-galleries/",
    "https://gem.eg/en/collection/main-galleries/",
    "https://gem.eg/en/collection/hanging-obelisk/",
    "https://gem.eg/en/collection/artefacts/the-golden-burial-mask-of-tutankhamun",
    "https://gem.eg/en/collection/artefacts/obelisk-of-king-ramesses-ii",
    "https://gem.eg/en/collection/artefacts/statuette-of-a-falcon",
    "https://gem.eg/en/collection/artefacts/model-of-funerary-boat-of-ukhhotep",
    "https://gem.eg/en/collection/artefacts/golden-throne",
    "https://gem.eg/en/collection/artefacts/royal-diadem",
    "https://gem.eg/en/collection/artefacts/colossal-statue-of-ramesses-ii",
    "https://gem.eg/en/collection/artefacts/colossus-of-a-ptolemaic-queen",
    "https://gem.eg/en/collection/artefacts/statue-of-god-ptah-king-ramesses-ii-and-sekhmet",
    "https://gem.eg/en/collection/artefacts/seated-statue-of-goddess-sekhmet",
    "https://gem.eg/en/collection/artefacts/column-of-king-merenptah",
    "https://gem.eg/en/collection/artefacts/hathor-capital",
    "https://gem.eg/en/collection/artefacts/pyramidion-of-an-obelisk-of-queen-hatshepsut",
    "https://gem.eg/en/collection/artefacts/fancy-perfume-vase",
    "https://gem.eg/en/collection/artefacts/mirror-of-mesehti",
    "https://gem.eg/en/collection/artefacts/necklace-with-ornaments-and-pendant",
    "https://gem.eg/en/collection/artefacts/a-gilt-hawk-figure-dendera-treasure-hoards",
    "https://gem.eg/en/collection/artefacts/colossus-of-a-ptolemaic-king-1",
    "https://gem.eg/en/research/conservation-centre/conservation-programmes/treatment-and-maintenance-of-archaeological-collections-and-science-of-museums/",
    "https://gem.eg/en/whats-on/events/al-mashrafia-2025/",
    "https://gem.eg/en/whats-on/events/riseup-summit-2025/",
    "https://gem.eg/en/whats-on/events/art-cairo-and-hiwar-programme/",
    "https://gem.eg/en/whats-on/events/um-kalthoum/",
    "https://gem.eg/en/whats-on/gem-programmes/gem-hackathon-third-edition/",
    # ── Arabic ────────────────────────────────
    "https://gem.eg/ar",
    "https://gem.eg/ar/visit",
    "https://gem.eg/ar/about",
    "https://gem.eg/ar/collection",
    "https://gem.eg/ar/visit/plan-your-visit/",
    "https://gem.eg/ar/visit/plan-your-visit/opening-hours/",
    "https://gem.eg/ar/visit/plan-your-visit/visitor-tips/",
    "https://gem.eg/ar/visit/plan-your-visit/museum-maps/",
    "https://gem.eg/ar/visit/accessibility/",
    "https://gem.eg/ar/gem-experience/arts-and-crafts-centre/",
    "https://gem.eg/ar/gem-experience/arts-and-crafts-centre/about-arts-and-crafts/",
    "https://gem.eg/ar/gem-experience/education-centre/",
    "https://gem.eg/ar/gem-experience/education-centre/about-the-centre/",
    "https://gem.eg/ar/gem-experience/children-museum/",
    "https://gem.eg/ar/gem-experience/children-museum/about-the-museum/",
    "https://gem.eg/ar/gem-experience/children-museum/cm-galleries/",
    "https://gem.eg/ar/collection/grand-hall/",
    "https://gem.eg/ar/collection/grand-stairs/",
    "https://gem.eg/ar/collection/tutankhamun-galleries/",
    "https://gem.eg/ar/collection/main-galleries/",
    "https://gem.eg/ar/collection/hanging-obelisk/",
    "https://gem.eg/ar/collection/artefacts/the-golden-burial-mask-of-tutankhamun",
    "https://gem.eg/ar/collection/artefacts/obelisk-of-king-ramesses-ii",
    "https://gem.eg/ar/collection/artefacts/statuette-of-a-falcon",
    "https://gem.eg/ar/collection/artefacts/model-of-funerary-boat-of-ukhhotep",
    "https://gem.eg/ar/collection/artefacts/golden-throne",
    "https://gem.eg/ar/collection/artefacts/royal-diadem",
    "https://gem.eg/ar/collection/artefacts/colossal-statue-of-ramesses-ii",
    "https://gem.eg/ar/collection/artefacts/colossus-of-a-ptolemaic-queen",
    "https://gem.eg/ar/collection/artefacts/statue-of-god-ptah-king-ramesses-ii-and-sekhmet",
    "https://gem.eg/ar/collection/artefacts/seated-statue-of-goddess-sekhmet",
    "https://gem.eg/ar/collection/artefacts/column-of-king-merenptah",
    "https://gem.eg/ar/collection/artefacts/hathor-capital",
    "https://gem.eg/ar/collection/artefacts/pyramidion-of-an-obelisk-of-queen-hatshepsut",
    "https://gem.eg/ar/collection/artefacts/fancy-perfume-vase",
    "https://gem.eg/ar/collection/artefacts/mirror-of-mesehti",
    "https://gem.eg/ar/collection/artefacts/necklace-with-ornaments-and-pendant",
    "https://gem.eg/ar/collection/artefacts/a-gilt-hawk-figure-dendera-treasure-hoards",
    "https://gem.eg/ar/collection/artefacts/colossus-of-a-ptolemaic-king-1",
    "https://gem.eg/ar/research/conservation-centre/conservation-programmes/treatment-and-maintenance-of-archaeological-collections-and-science-of-museums/",
    "https://gem.eg/ar/whats-on/events/al-mashrafia-2025/",
    "https://gem.eg/ar/whats-on/events/riseup-summit-2025/",
    "https://gem.eg/ar/whats-on/events/art-cairo-and-hiwar-programme/",
    "https://gem.eg/ar/whats-on/events/um-kalthoum/",
    "https://gem.eg/ar/whats-on/gem-programmes/gem-hackathon-third-edition/",
]

EXCLUDED_URL_PATTERNS = [
    "/terms-and-conditions",
    "/privacy-policy",
    "/cookie-policy",
]

MAX_RETRIES       = 2
CHECKPOINT_EVERY  = 10      # save partial results every N URLs
MIN_CONTENT_LEN   = {"ar": 50, "en": 80}


# ─────────────────────────────────────────────
# Shared Utilities (inlined — no separate file needed in Modal)
# ─────────────────────────────────────────────
def detect_language(url: str) -> str:
    """Detect language from URL prefix — most reliable signal."""
    from urllib.parse import urlparse
    path = urlparse(url).path
    return "ar" if (path.startswith("/ar") or path.startswith("/ar/")) else "en"


def is_arabic_text(text: str, threshold: float = 0.2) -> bool:
    """Return True if >threshold fraction of chars are Arabic Unicode."""
    if not text:
        return False
    arabic = sum(
        1 for c in text
        if "\u0600" <= c <= "\u06ff"
        or "\u0750" <= c <= "\u077f"
        or "\ufb50" <= c <= "\ufdff"
        or "\ufe70" <= c <= "\ufeff"
    )
    return arabic / max(len(text), 1) > threshold


def categorize_url(url: str) -> str:
    url_lower = url.lower()
    cats = {
        "collection":   ["collection", "artefact", "artifact"],
        "visitor_info": ["visit", "ticket", "hour", "plan", "map", "tip", "access"],
        "educational":  ["education", "children", "arts-and-crafts"],
        "events":       ["event", "whats-on", "programme", "hackathon"],
        "about":        ["about", "research", "conservation"],
        "exhibition":   ["exhibition", "gallery", "galleries", "hall", "stairs"],
    }
    for cat, kws in cats.items():
        if any(kw in url_lower for kw in kws):
            return cat
    return "general"


def is_excluded_url(url: str) -> bool:
    return any(p in url for p in EXCLUDED_URL_PATTERNS)


def is_error_page(title: str, url: str) -> bool:
    if "error-page" in url or "aspxerrorpath" in url:
        return True
    return any(x in title for x in ["500", "404", "Error", "Not Found"])


# ─────────────────────────────────────────────
# Modal Function
# ─────────────────────────────────────────────
@app.function(
    image=scraper_image,
    volumes={VOLUME_PATH: volume},
    timeout=3600,
    memory=4096,
)
def run_scraper():
    """
    Scrapes gem.eg using Playwright.
    - Fresh page per URL for crash isolation
    - Retry with exponential backoff
    - Incremental checkpointing every CHECKPOINT_EVERY URLs
    - Saves all_documents.json to Modal Volume 'rag'
    """

    import json
    import time
    import random
    import logging
    from pathlib import Path
    from datetime import datetime
    from collections import Counter
    from urllib.parse import urlparse

    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    from bs4 import BeautifulSoup

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # ─────────────────────────────────────────
    # Content Extractor
    # ─────────────────────────────────────────
    def extract_content(html: str, language: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")

        # Remove structural noise — block tags
        for tag in ["script", "style", "noscript", "iframe",
                    "svg", "form", "header", "nav", "footer"]:
            for el in soup.find_all(tag):
                el.decompose()

        # Remove noise by CSS class keywords (CSS selector — faster than lambda)
        noise_keywords = [
            "cookie", "popup", "modal", "overlay", "social",
            "share", "newsletter", "advertisement", "banner",
            "breadcrumb", "pagination", "sidebar", "related",
        ]
        for kw in noise_keywords:
            # Targets any element whose class attribute contains the keyword
            for el in soup.select(f'[class*="{kw}"]'):
                el.decompose()

        # ── Title extraction ──────────────────
        title = "Grand Egyptian Museum"
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            title = og["content"].strip()
        elif soup.title:
            raw = soup.title.get_text(strip=True)
            for sep in [" - ", " | ", " – ", " — "]:
                if sep in raw:
                    parts = [p.strip() for p in raw.split(sep)]
                    museum_names = {
                        "Grand Egyptian Museum",
                        "المتحف المصري الكبير",
                    }
                    chosen = next(
                        (p for p in parts if p and p not in museum_names),
                        parts[0],
                    )
                    title = chosen
                    break
            else:
                title = raw

        # ── Main content area ─────────────────
        main_area = (
            soup.select_one("main")
            or soup.select_one('[role="main"]')
            or soup.select_one(".main-content")
            or soup.select_one("#main-content")
            or soup.select_one(".page-content")
            or soup.select_one(".content-wrapper")
            or soup.select_one(".container")
            or soup.find("body")
            or soup
        )

        # ── Text extraction ───────────────────
        parts: list[str] = []
        seen:  set[str]  = set()
        block_tags = {"p", "h1", "h2", "h3", "h4", "h5", "h6",
                      "ul", "ol", "li"}

        for el in main_area.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6",
             "p", "li", "td", "th", "div", "span"]
        ):
            # Skip div/span that are containers (have block children)
            if el.name in {"div", "span"}:
                if any(
                    getattr(child, "name", None) in block_tags
                    for child in el.children
                ):
                    continue

            text = " ".join(el.get_text().split()).strip()
            if not text or text in seen:
                continue

            min_len = 8 if language == "ar" else 15
            if len(text) < min_len:
                continue

            seen.add(text)
            parts.append(text)

        return {"title": title, "content": "\n\n".join(parts)}

    # ─────────────────────────────────────────
    # Resume from checkpoint if it exists
    # ─────────────────────────────────────────
    partial_path  = Path(VOLUME_PATH) / "all_documents_partial.json"
    scraped_urls: set[str] = set()
    documents: list[dict]  = []

    if partial_path.exists():
        try:
            with open(partial_path, "r", encoding="utf-8") as f:
                documents = json.load(f)
            scraped_urls = {d["url"] for d in documents}
            logger.info(
                f"♻️  Resuming — {len(documents)} docs already scraped"
            )
        except Exception:
            documents    = []
            scraped_urls = set()

    # ─────────────────────────────────────────
    # Deduplicate seeds (preserve order)
    # ─────────────────────────────────────────
    deduped_seeds = list(dict.fromkeys(SEED_URLS))
    remaining     = [u for u in deduped_seeds if u not in scraped_urls
                     and not is_excluded_url(u)]

    skipped_urls: list[str] = [
        u for u in deduped_seeds if is_excluded_url(u)
    ]
    failed_urls:  list[dict] = []

    print(f"\n🏛️  GEM Playwright Scraper")
    print(f"📄 Total seeds    : {len(deduped_seeds)}")
    print(f"✅ Already done   : {len(scraped_urls)}")
    print(f"🔄 To scrape      : {len(remaining)}")
    print(f"⛔ Excluded       : {len(skipped_urls)}")
    print("=" * 60)

    # ─────────────────────────────────────────
    # Playwright session
    # ─────────────────────────────────────────
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-blink-features=AutomationControlled",
                "--disable-extensions",
                "--no-first-run",
                "--no-zygote",
                "--single-process",
            ],
        )

        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="Africa/Cairo",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9,ar;q=0.8"},
        )

        # Mask webdriver flag
        context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)

        total = len(remaining)

        for i, url in enumerate(remaining):
            language = detect_language(url)
            category = categorize_url(url)
            label    = f"[{i+1}/{total}] [{language.upper()}]"

            print(f"\n{label} {url}")

            # ── Per-URL page (crash isolation) ──
            page    = context.new_page()
            success = False

            for attempt in range(MAX_RETRIES + 1):
                try:
                    # networkidle catches JS-rendered content better
                    page.goto(
                        url,
                        wait_until="networkidle",
                        timeout=60_000,
                    )
                    page.wait_for_timeout(2_000)
                    page.evaluate(
                        "window.scrollTo(0, document.body.scrollHeight)"
                    )
                    page.wait_for_timeout(1_500)
                    page.evaluate("window.scrollTo(0, 0)")
                    page.wait_for_timeout(500)

                    html      = page.content()
                    extracted = extract_content(html, language)
                    title     = extracted["title"]
                    content   = extracted["content"]

                    if is_error_page(title, page.url):
                        print(f"  ⚠️  Error page detected — skipping")
                        failed_urls.append({"url": url, "reason": "error_page"})
                        break

                    min_len = MIN_CONTENT_LEN.get(language, 80)
                    if len(content) < min_len:
                        print(
                            f"  ⚠️  Too short ({len(content)} chars) — skipping"
                        )
                        failed_urls.append({
                            "url":    url,
                            "reason": f"too_short ({len(content)} chars)",
                        })
                        break

                    documents.append({
                        "title":          title,
                        "url":            url,
                        "category":       category,
                        "language":       language,
                        "date":           datetime.now().strftime("%Y-%m-%d"),
                        "content":        content,
                        "content_length": len(content),
                        "extracted_at":   datetime.now().isoformat(),
                    })

                    print(f"  Title   : {title[:60]}")
                    print(f"  Content : {len(content):,} chars  ✅")
                    success = True
                    break   # exit retry loop

                except PWTimeout:
                    wait = 2 ** attempt
                    print(
                        f"  ⏱  Timeout (attempt {attempt+1}/{MAX_RETRIES+1}) "
                        f"— retrying in {wait}s"
                    )
                    time.sleep(wait)

                except Exception as exc:
                    wait = 2 ** attempt
                    print(
                        f"  ❌ Error (attempt {attempt+1}/{MAX_RETRIES+1}): "
                        f"{exc} — retrying in {wait}s"
                    )
                    time.sleep(wait)

            else:
                # All retries exhausted
                failed_urls.append({
                    "url":    url,
                    "reason": "max_retries_exceeded",
                })

            page.close()     # always close the per-URL page

            # ── Checkpoint every N URLs ──────────
            if (i + 1) % CHECKPOINT_EVERY == 0:
                with open(partial_path, "w", encoding="utf-8") as f:
                    json.dump(documents, f, ensure_ascii=False)
                volume.commit()
                print(
                    f"\n  💾 Checkpoint saved "
                    f"({len(documents)} docs so far)"
                )

            # Polite delay
            time.sleep(random.uniform(1.5, 3.0))

        browser.close()

    # ─────────────────────────────────────────
    # Final save
    # ─────────────────────────────────────────
    output_path = Path(VOLUME_PATH) / "all_documents.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    # Remove partial checkpoint — full file is now written
    if partial_path.exists():
        partial_path.unlink()

    volume.commit()

    # ─────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────
    lang_counts = Counter(d["language"] for d in documents)
    cat_counts  = Counter(d["category"] for d in documents)

    print(f"\n{'='*60}")
    print(f"✅ SCRAPING COMPLETE")
    print(f"{'='*60}")
    print(f"📄 Total documents : {len(documents)}")
    print(f"🌐 English         : {lang_counts.get('en', 0)}")
    print(f"🌐 Arabic          : {lang_counts.get('ar', 0)}")
    print(f"\n📊 By Category:")
    for cat, count in cat_counts.most_common():
        print(f"   {cat:<20}: {count}")
    if failed_urls:
        print(f"\n⚠️  Failed ({len(failed_urls)}):")
        for item in failed_urls:
            print(f"   ❌ {item['url']}")
            print(f"      Reason: {item['reason']}")
    else:
        print("\n✅ No failures!")

    if documents:
        best = max(documents, key=lambda d: d["content_length"])
        print(f"\n📖 Largest document:")
        print(f"   Title  : {best['title']}")
        print(f"   Length : {best['content_length']:,} chars")
        print(f"   Preview: {best['content'][:200]}...")

    print(f"\n💾 Saved → {output_path}")

    return {
        "total_documents": len(documents),
        "english":         lang_counts.get("en", 0),
        "arabic":          lang_counts.get("ar", 0),
        "failed":          len(failed_urls),
        "excluded":        len(skipped_urls),
        "saved_to":        str(output_path),
    }


# ─────────────────────────────────────────────
# Local Entrypoint
# ─────────────────────────────────────────────
@app.local_entrypoint()
def main():
    print("🚀 Launching GEM Scraper on Modal...")
    result = run_scraper.remote()
    print(f"\n🎉 Done!")
    print(f"   Total : {result['total_documents']}")
    print(f"   EN    : {result['english']}")
    print(f"   AR    : {result['arabic']}")
    print(f"   Failed: {result['failed']}")