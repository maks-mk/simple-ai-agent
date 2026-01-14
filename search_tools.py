import asyncio
import httpx
import re
import urllib.parse
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List

from bs4 import BeautifulSoup
from langchain_core.tools import tool


# ======================================================
# RATE LIMITER
# ======================================================

class RateLimiter:
    """
    Simple async rate limiter (requests per minute).
    """

    def __init__(self, rpm: int = 30):
        self.rpm = rpm
        self.calls: List[datetime] = []

    async def acquire(self):
        now = datetime.now()
        self.calls = [t for t in self.calls if now - t < timedelta(minutes=1)]

        if len(self.calls) >= self.rpm:
            wait_time = 60 - (now - self.calls[0]).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.calls.append(now)


# ======================================================
# DATA STRUCTURES
# ======================================================

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    position: int


# ======================================================
# DUCKDUCKGO SEARCH CLIENT
# ======================================================

class DuckDuckGoClient:
    """
    DuckDuckGo HTML search.
    Imitates a browser to bypass basic bot protection.
    """

    BASE_URL = "https://html.duckduckgo.com/html"
    
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://duckduckgo.com/"
    }

    def __init__(self, rpm: int = 20):
        self.limiter = RateLimiter(rpm=rpm)

    async def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        await self.limiter.acquire()

        try:
            async with httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers=self.HEADERS
            ) as client:
                response = await client.post(
                    self.BASE_URL,
                    data={"q": query, "b": "", "kl": ""},
                )
                response.raise_for_status()
        except Exception:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        results: List[SearchResult] = []

        for block in soup.select(".result"):
            title_el = block.select_one(".result__a")
            snippet_el = block.select_one(".result__snippet")

            if not title_el:
                continue

            url = title_el.get("href", "")
            if not url:
                continue

            if "y.js" in url or "duckduckgo.com" in url:
                continue

            if "/l/?uddg=" in url:
                try:
                    url = urllib.parse.unquote(
                        url.split("uddg=")[1].split("&")[0]
                    )
                except IndexError:
                    continue

            results.append(
                SearchResult(
                    title=title_el.get_text(strip=True),
                    url=url,
                    snippet=snippet_el.get_text(strip=True) if snippet_el else "",
                    position=len(results) + 1,
                )
            )

            if len(results) >= limit:
                break

        return results


# ======================================================
# WEB PAGE FETCHER
# ======================================================

class WebFetcher:
    """
    Fetches web pages with improved URL validation.
    Resilient to JSON input and argument garbage.
    """

    def __init__(self, rpm: int = 20, max_chars: int = 8000):
        self.limiter = RateLimiter(rpm=rpm)
        self.max_chars = max_chars

    async def fetch(self, raw_input: str) -> str:
        raw_input = str(raw_input).strip()
        target_url = None

        # 1. Try to extract URL from JSON-like structure
        if "{" in raw_input and "}" in raw_input:
            try:
                json_match = re.search(r'(\{.*\})', raw_input, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(1))
                    if isinstance(data, dict):
                        for v in data.values():
                            if isinstance(v, str) and v.startswith("http"):
                                target_url = v
                                break
            except Exception:
                pass

        # 2. If no URL in JSON, look for http/https regex
        if not target_url:
            match = re.search(r'(https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s\'"<>\[\]{}()]*)', raw_input)
            if match:
                target_url = match.group(1)
                target_url = target_url.rstrip(".,;:)'\"")

        # 3. Fallback: clean the string
        if not target_url:
            clean = re.sub(r'[\s\'"<>\[\]{}():,]+', '', raw_input).replace("url=", "")
            if "." in clean and len(clean) > 4:
                target_url = f"https://{clean}"

        # 4. Final Validation (English Error Message)
        if not target_url or not target_url.startswith("http") or len(target_url) < 8:
            return f"Error: Invalid URL input '{raw_input}'. Please provide a valid absolute URL starting with http:// or https://."

        await self.limiter.acquire()

        try:
            async with httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
                },
                verify=False 
            ) as client:
                response = await client.get(target_url)
                
                if response.status_code == 404:
                    return f"Error 404: Page not found ({target_url})"
                if response.status_code in [403, 401]:
                    return f"Error {response.status_code}: Access denied (bot protection) ({target_url})"
                
                response.raise_for_status()

        except Exception as e:
            return f"System Error loading {target_url}: {str(e)}"

        soup = BeautifulSoup(response.text, "html.parser")

        # Cleanup
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "iframe", "noscript", "svg"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) > self.max_chars:
            text = text[: self.max_chars] + " …[truncated]"
        
        if not text:
            return "Page loaded, but no text content found (possibly SPA/React site)."

        return text


# ======================================================
# SINGLETONS
# ======================================================

_ddg = DuckDuckGoClient()
_fetcher = WebFetcher()


# ======================================================
# TOOLS
# ======================================================

@tool
async def web_search(query: str) -> str:
    """
    Search for information online. Returns a list of links and snippets.
    """
    results = await _ddg.search(query, limit=5)

    if not results:
        return "No results found (try rephrasing the query)."

    return "\n\n".join(
        f"{r.position}. {r.title}\n{r.url}\n{r.snippet}"
        for r in results
    )


@tool
async def fetch_url(url: str) -> str:
    """
    Fetches and returns the text content of any web page by URL.
    Input must be a URL string.
    """
    return await _fetcher.fetch(url)