"""
local_crawler.py
Run this LOCALLY on your machine
Saves crawled data to local folder
then upload to Modal volume
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import json
import logging
from typing import Set, List, Dict, Optional
from pathlib import Path
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GEMCrawler:

    def __init__(self, base_url: str = "https://gem.gov.eg"):
        self.base_url     = base_url
        self.visited_urls: Set[str] = set()
        self.collected_urls: List[Dict] = []

        self.target_sections = [
            # English
            "/en/exhibitions",
            "/en/collections",
            "/en/visitor-information",
            "/en/educational",
            "/en/news",
            "/en/events",
            "/en/about",
            "/en/faqs",
            "/en/blog",
            # Arabic
            "/ar/exhibitions",
            "/ar/collections",
            "/ar/visitor-information",
            "/ar/educational",
            "/ar/news",
            "/ar/events",
            "/ar/about",
            "/ar/faqs",
            "/ar/blog",
        ]

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
            "Accept-Charset": "utf-8",
        }

        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def detect_language(self, url: str, soup: BeautifulSoup) -> str:
        if "/ar/" in url:
            return "ar"
        if "/en/" in url:
            return "en"
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            lang = html_tag["lang"].lower()
            if lang.startswith("ar"):
                return "ar"
        body = soup.find("body")
        if body:
            text_sample = body.get_text()[:500]
            arabic_chars = sum(
                1 for char in text_sample
                if "\u0600" <= char <= "\u06ff"
            )
            if arabic_chars > 20:
                return "ar"
        return "en"

    def is_valid_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.netloc and parsed.netloc != urlparse(self.base_url).netloc:
            return False
        skip_ext = [
            '.pdf', '.jpg', '.jpeg', '.png', '.gif',
            '.css', '.js', '.ico', '.svg', '.woff', '.woff2'
        ]
        if any(url.lower().endswith(ext) for ext in skip_ext):
            return False
        skip_domains = ['facebook', 'twitter', 'instagram', 'youtube', 'linkedin']
        if any(d in url.lower() for d in skip_domains):
            return False
        return True

    def categorize_url(self, url: str) -> str:
        url_lower = url.lower()
        categories = {
            "exhibition":   ["exhibition", "exhibit", "معرض"],
            "collection":   ["collection", "artifact", "antiquit", "مجموعة"],
            "visitor_info": ["visitor", "hours", "ticket", "plan", "access", "زائر", "تذكرة", "مواعيد"],
            "news":         ["news", "press", "media", "اخبار", "أخبار"],
            "events":       ["event", "program", "workshop", "فعالية", "برنامج"],
            "educational":  ["education", "learn", "school", "student", "تعليم", "مدرسة"],
            "about":        ["about", "history", "mission", "عن", "تاريخ"],
            "faq":          ["faq", "question", "help", "أسئلة"],
            "blog":         ["blog", "article", "story", "مقال"],
        }
        for category, keywords in categories.items():
            if any(kw in url_lower for kw in keywords):
                return category
        return "general"

    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        try:
            logger.info(f"Fetching: {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            response.encoding = "utf-8"
            time.sleep(2)   # Be respectful
            return BeautifulSoup(response.text, "lxml")
        except Exception as e:
            logger.error(f"Failed: {url} → {e}")
            return None

    def extract_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        links = []
        for anchor in soup.find_all("a", href=True):
            full_url = urljoin(current_url, anchor["href"]).split("#")[0]
            if self.is_valid_url(full_url) and full_url not in self.visited_urls:
                links.append(full_url)
        return links

    def _extract_title(self, soup: BeautifulSoup) -> str:
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            return og["content"].strip()
        if soup.title:
            title = soup.title.get_text(strip=True)
            for sep in ["|", "–", "-"]:
                if sep in title:
                    return title.split(sep)[0].strip()
            return title
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return "Unknown"

    def crawl(self, max_pages: int = 300) -> List[Dict]:
        urls_to_visit = [self.base_url] + [
            self.base_url + s for s in self.target_sections
        ]
        pages_crawled = 0

        while urls_to_visit and pages_crawled < max_pages:
            current_url = urls_to_visit.pop(0)
            if current_url in self.visited_urls:
                continue
            self.visited_urls.add(current_url)

            soup = self.fetch_page(current_url)
            if not soup:
                continue

            language = self.detect_language(current_url, soup)

            self.collected_urls.append({
                "url":      current_url,
                "category": self.categorize_url(current_url),
                "title":    self._extract_title(soup),
                "language": language,
            })
            pages_crawled += 1
            logger.info(
                f"[{language.upper()}] {pages_crawled}/{max_pages}: {current_url}"
            )

            new_links = self.extract_links(soup, current_url)
            urls_to_visit.extend(new_links)

        return self.collected_urls

    def save(self, output_path: str = "data/crawled_urls.json"):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.collected_urls, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.collected_urls)} URLs → {output_path}")


# ─────────────────────────────────────────────
# Run Locally
# ─────────────────────────────────────────────
if __name__ == "__main__":
    crawler = GEMCrawler()
    pages   = crawler.crawl(max_pages=300)
    crawler.save("data/crawled_urls.json")

    lang_counts = Counter(p["language"] for p in pages)
    categories  = Counter(p["category"] for p in pages)

    print(f"\n✅ Done! Total pages: {len(pages)}")
    print(f"🌐 English: {lang_counts.get('en', 0)}")
    print(f"🌐 Arabic : {lang_counts.get('ar', 0)}")
    print(f"\n📊 Categories:")
    for cat, count in categories.most_common():
        print(f"   {cat}: {count}")