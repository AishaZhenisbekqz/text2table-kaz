"""
Asynchronous web crawler for Egemen Qazaqstan archive.

Implements the data acquisition pipeline described in Section III-A of the paper.
Uses aiohttp + BeautifulSoup with exponential-backoff retry (b=2, c=10).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import AsyncIterator

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://egemen.kz"
ARCHIVE_URL = f"{BASE_URL}/archiv"

# Exponential backoff parameters (paper, Section III-A)
BACKOFF_BASE = 2
BACKOFF_MAX_RETRIES = 10
CONCURRENT_REQUESTS = 8


class EgemenCrawler:
    """
    Asynchronous crawler for Egemen Qazaqstan.

    Traverses the archive from start_date to end_date,
    extracts article metadata and full text, and saves
    each article as a JSON record.

    Output fields per record:
        url, title, date, author, abstract, text, journal, category
    """

    def __init__(
        self,
        output_dir: str | Path = "data/raw",
        start_date: date = date(2017, 1, 1),
        end_date: date = date(2025, 12, 31),
        concurrent: int = CONCURRENT_REQUESTS,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_date = start_date
        self.end_date = end_date
        self.concurrent = concurrent
        self._semaphore: asyncio.Semaphore | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Synchronous entry point — runs the async crawler."""
        asyncio.run(self._crawl_all())

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    async def _crawl_all(self) -> None:
        self._semaphore = asyncio.Semaphore(self.concurrent)
        connector = aiohttp.TCPConnector(limit=self.concurrent)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            dates = list(self._date_range(self.start_date, self.end_date))
            tasks = [self._crawl_day(session, d) for d in dates]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        total = sum(r for r in results if isinstance(r, int))
        logger.info(f"Crawl complete. Total articles: {total}")

    async def _crawl_day(
        self, session: aiohttp.ClientSession, day: date
    ) -> int:
        """Crawl all articles for a given date. Returns article count."""
        async with self._semaphore:
            url = f"{ARCHIVE_URL}/{day.strftime('%Y/%m/%d')}"
            html = await self._fetch(session, url)
            if html is None:
                return 0

            soup = BeautifulSoup(html, "lxml")
            article_links = self._extract_article_links(soup)
            count = 0

            for link in article_links:
                article = await self._fetch_article(session, link, day)
                if article:
                    self._save(article)
                    count += 1

            return count

    async def _fetch_article(
        self,
        session: aiohttp.ClientSession,
        url: str,
        day: date,
    ) -> dict | None:
        html = await self._fetch(session, url)
        if html is None:
            return None
        return self._parse_article(html, url, day)

    async def _fetch(
        self, session: aiohttp.ClientSession, url: str
    ) -> str | None:
        """Fetch URL with exponential-backoff retry."""
        for attempt in range(BACKOFF_MAX_RETRIES):
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        return await resp.text(encoding="utf-8", errors="replace")
                    logger.warning(f"HTTP {resp.status} for {url}")
                    return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                wait = BACKOFF_BASE ** attempt
                logger.debug(f"Retry {attempt+1}/{BACKOFF_MAX_RETRIES} for {url}: {exc}")
                await asyncio.sleep(min(wait, 60))

        logger.error(f"Failed after {BACKOFF_MAX_RETRIES} retries: {url}")
        return None

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_article_links(soup: BeautifulSoup) -> list[str]:
        """Extract article URLs from archive listing page."""
        links = []
        for tag in soup.select("a.article-link, h2.entry-title a, .news-title a"):
            href = tag.get("href", "")
            if href and not href.startswith("http"):
                href = BASE_URL + href
            if href:
                links.append(href)
        return list(set(links))

    @staticmethod
    def _parse_article(html: str, url: str, day: date) -> dict | None:
        """Parse full article page into structured record."""
        soup = BeautifulSoup(html, "lxml")

        title_tag = soup.find("h1", class_=lambda c: c and "title" in c)
        title = title_tag.get_text(strip=True) if title_tag else ""

        author_tag = soup.find(class_=lambda c: c and "author" in str(c))
        author = author_tag.get_text(strip=True) if author_tag else ""

        # Remove navigation, ads, pagination
        for tag in soup.select("nav, .advertisement, .pagination, footer, script, style"):
            tag.decompose()

        body_tag = soup.find("div", class_=lambda c: c and "content" in str(c))
        if body_tag is None:
            body_tag = soup.find("article")
        text = body_tag.get_text(separator="\n", strip=True) if body_tag else ""

        if not text or len(text) < 100:
            return None

        category_tag = soup.find(class_=lambda c: c and "category" in str(c))
        category = category_tag.get_text(strip=True) if category_tag else ""

        abstract_tag = soup.find("meta", attrs={"name": "description"})
        abstract = abstract_tag.get("content", "") if abstract_tag else ""

        return {
            "url": url,
            "title": title,
            "date": day.isoformat(),
            "author": author,
            "abstract": abstract,
            "text": text,
            "journal": "Egemen Qazaqstan",
            "category": category,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _save(self, article: dict) -> None:
        day = article["date"]
        out_file = self.output_dir / f"{day}.jsonl"
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    @staticmethod
    def _date_range(start: date, end: date) -> list[date]:
        days = []
        current = start
        while current <= end:
            days.append(current)
            current += timedelta(days=1)
        return days
