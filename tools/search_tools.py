import os
import logging
import asyncio
import json
import hashlib
import time
from functools import wraps
from typing import Optional, List, Any, Union, Dict

from langchain_core.tools import tool

from core.config import AgentConfig

logger = logging.getLogger(__name__)

# --- Инициализация Tavily ---
try:
    from tavily import AsyncTavilyClient
except ImportError:
    AsyncTavilyClient = None
    logger.warning("Tavily SDK not installed. Search tools will be disabled.")

_client: Optional[Any] = None
# Семафор для ограничения параллельных запросов (Rate Limit)
_search_semaphore = asyncio.Semaphore(5)

# ======================================================
# CACHING SYSTEM (In-Memory TTL + Size Limit)
# ======================================================

_SEARCH_CACHE: Dict[str, tuple[Any, float]] = {}
_MAX_CACHE_SIZE = 50

def _cleanup_cache():
    """Удаляет устаревшие или лишние записи, если кэш переполнен."""
    if len(_SEARCH_CACHE) <= _MAX_CACHE_SIZE:
        return

    # Сортируем по времени (старые в начале)
    sorted_items = sorted(_SEARCH_CACHE.items(), key=lambda item: item[1][1])
    
    # Удаляем 20% самых старых записей
    remove_count = int(_MAX_CACHE_SIZE * 0.2) + 1
    for k, _ in sorted_items[:remove_count]:
        del _SEARCH_CACHE[k]
    
    logger.debug(f"🧹 Cache cleanup: removed {remove_count} items. Size: {len(_SEARCH_CACHE)}")

def with_cache(ttl: int = 600):
    """
    Асинхронный декоратор для кэширования результатов (Time-To-Live).
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                # Сериализуем аргументы в JSON для хэширования
                key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
                key = hashlib.md5(key_data.encode()).hexdigest()
            except Exception:
                return await func(*args, **kwargs)

            # Проверка кэша
            if key in _SEARCH_CACHE:
                result, timestamp = _SEARCH_CACHE[key]
                if time.time() - timestamp < ttl:
                    logger.info(f"⚡ Cache hit for {func.__name__} (key: {key[:8]})")
                    return result
                else:
                    del _SEARCH_CACHE[key]

            # Выполнение
            result = await func(*args, **kwargs)

            # Сохранение (если не ошибка)
            if isinstance(result, str) and not result.lower().startswith(("error:", "ошибка:")):
                if len(_SEARCH_CACHE) >= _MAX_CACHE_SIZE:
                    _cleanup_cache()
                _SEARCH_CACHE[key] = (result, time.time())
            
            return result
        return wrapper
    return decorator

# ======================================================
# HELPERS
# ======================================================

def _format_error(error_type: str, details: str = "") -> str:
    """Унифицированное форматирование ошибок."""
    error_templates = {
        "disabled": "Error: Search tools are disabled by configuration (ENABLE_SEARCH_TOOLS=False).",
        "missing_config": "Error: Search is unavailable due to missing configuration (API Key or SDK).",
        "empty_query": "Error: Empty query provided.",
        "api_error": "Error: Search failed after 3 attempts. Details: {details}",
        "auth_error": "Error: Invalid API credentials (401 Unauthorized).",
        "no_results": "Error: No results found.",
    }
    msg = error_templates.get(error_type, f"Error: {details}")
    if "{details}" in msg:
        msg = msg.format(details=details)
    return msg

def get_tavily_client() -> Optional[Any]:
    global _client
    
    # Load config once or per call (lightweight)
    try:
        config = AgentConfig()
    except Exception as e:
        logger.error(f"Config load failed: {e}")
        return None

    if not config.enable_search_tools:
        return None 

    if _client is not None:
        return _client

    if AsyncTavilyClient is None:
        return None

    if not config.tavily_api_key:
        logger.error("TAVILY_API_KEY is not set.")
        return None

    try:
        _client = AsyncTavilyClient(api_key=config.tavily_api_key.get_secret_value())
        return _client
    except Exception as e:
        logger.error(f"Failed to initialize Tavily client: {e}")
        return None

async def _execute_with_retry(coroutine_func, *args, **kwargs):
    """Выполняет запрос с ретраями и семафором."""
    try:
        config = AgentConfig()
        max_retries = config.max_retries
        retry_delay = config.retry_delay
    except Exception:
        max_retries = 3
        retry_delay = 2

    last_error = None

    async with _search_semaphore:
        for attempt in range(max_retries):
            try:
                return await coroutine_func(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Search attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
    raise last_error

# ======================================================
# TOOLS
# ======================================================

@tool("web_search")
@with_cache(ttl=600)
async def web_search(query: str, max_results: int = 5) -> str:
    """
    Search internet for snippets and AI summary. Best for facts, news, comparisons.
    """
    if not get_tavily_client():
        return _format_error("missing_config")

    query = (query or "").strip()
    if not query:
        return _format_error("empty_query")

    client = get_tavily_client()
    try:
        response = await _execute_with_retry(
            client.search,
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=True
        )
    except Exception as e:
        msg = str(e).lower()
        if "401" in msg or "unauthorized" in msg:
            return _format_error("auth_error")
        return _format_error("api_error", details=str(e))

    results = []
    
    # AI Answer
    answer = response.get("answer")
    if answer:
        results.append(f"AI Overview:\n{answer}\n{'='*40}")

    items = response.get("results", [])
    if not items:
        return "\n".join(results) if results else _format_error("no_results")

    # Format compact result
    total_chars = 0
    max_chars = 11000 
    separator = "-" * 40

    for item in items:
        title = item.get("title") or "Untitled"
        url = item.get("url") or ""
        content = item.get("content") or ""
        score = item.get("score", 0)

        if not content: continue

        header = f"Source: {title} (score: {score:.2f})\nURL: {url}"
        block = f"{header}\n{content}"
        
        if total_chars + len(block) > max_chars:
            break
            
        results.append(block)
        results.append(separator)
        total_chars += len(block) + len(separator)

    results.append("[Search completed. Use the context above.]")
    return "\n".join(results)

@tool("fetch_content")
@with_cache(ttl=1800)
async def fetch_content(urls: Union[str, List[str]], advanced: bool = False) -> str:
    """
    Extract text from one or multiple URLs. 
    Use this to read pages after searching. Supports batching (up to 20 links).
    """
    client = get_tavily_client()
    if not client:
        return "Error: Fetch unavailable due to missing configuration."

    if isinstance(urls, str):
        urls = [urls]
    
    clean_urls = []
    for u in urls:
        if isinstance(u, str) and u.startswith("http"):
             clean_urls.append(u.strip().strip('"\''))
    
    clean_urls = list(set(clean_urls))[:20]
    if not clean_urls:
        return "Error: No valid URLs provided."

    depth = "advanced" if advanced else "basic"
    try:
        response = await _execute_with_retry(client.extract, urls=clean_urls, extract_depth=depth)
    except Exception as e:
        return f"Error: Fetch failed. Details: {e}"

    output_parts = []
    max_chars_per_url = 8000 if len(clean_urls) == 1 else 3000
    
    for item in response.get("results", []):
        url = item.get("url", "Unknown")
        content = item.get("raw_content") or item.get("content") or ""
        
        if len(content) > max_chars_per_url:
            content = content[:max_chars_per_url] + "\n... [Truncated]"
        
        output_parts.append(f"=== SOURCE: {url} ===\n{content or '[Empty]'}\n{'='*30}")

    failed = response.get("failed_results", [])
    if failed:
        for f in failed:
            output_parts.append(f"❌ FAILED: {f.get('url')} - {f.get('error')}")

    return "\n".join(output_parts) or "Error: No content extracted."

@tool("batch_web_search")
@with_cache(ttl=600)
async def batch_web_search(queries: List[str]) -> str:
    """
    Runs multiple queries in parallel. Much faster/cheaper than calling web_search N times.
    Use for: multi-aspect research, verifying facts from different angles.
    """
    client = get_tavily_client()
    if not client:
        return "Error: Search unavailable."

    queries = queries[:5]
    tasks = [
        client.search(query=q, max_results=4, search_depth="basic", include_answer=True)
        for q in queries
    ]

    try:
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        return f"Batch search error: {e}"

    output = []
    for q, res in zip(queries, results_list):
        output.append(f"\n=== Search: {q} ===")
        if isinstance(res, Exception):
            output.append(f"Error: {res}")
            continue
            
        if res.get("answer"):
            output.append(f"AI Summary: {res.get('answer')}")
            
        for item in res.get("results", []):
            title = item.get('title', 'No Title')
            url = item.get('url', 'No URL')
            content = item.get('content', '')[:200]
            output.append(f"- {title} ({url}): {content}...")
            
    return "\n".join(output)

@tool("crawl_site")
@with_cache(ttl=3600)
async def crawl_site(
    start_url: str,
    max_depth: int = 3,
    limit: int = 50,
    include_subdomains: bool = False,) -> str:
    """
    Crawl a website starting from start_url and collect relevant pages.
    """
    client = get_tavily_client()
    if not client: return "Error: Config missing."

    try:
        response = await _execute_with_retry(
            client.crawl,
            url=start_url,
            max_depth=max_depth,
            limit=limit,
            include_subdomains=include_subdomains
        )
    except Exception as e:
        return f"Crawl failed: {e}"

    pages = response.get("results", []) or response.get("pages", [])
    if not pages: return "No pages crawled."

    parts = []
    for idx, page in enumerate(pages[:20], 1): # Limit output size
        content = page.get("raw_content") or page.get("content") or ""
        if content:
            parts.append(f"[Page {idx}] {page.get('url')}\n{'='*40}\n{content[:2000]}\n")

    return "\n".join(parts)
