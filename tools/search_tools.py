import os
import logging
import asyncio
import json
import hashlib
import time
from functools import wraps, lru_cache
from typing import Optional, List, Any, Union, Dict

from langchain_core.tools import tool

from core.config import AgentConfig
from core.safety_policy import SafetyPolicy
from core.errors import format_error, ErrorType
from core.utils import truncate_output

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

# Global safety policy
_SAFETY_POLICY: Optional[SafetyPolicy] = None

def set_safety_policy(policy: SafetyPolicy):
    global _SAFETY_POLICY
    _SAFETY_POLICY = policy

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

@lru_cache(maxsize=1)
def _get_config() -> AgentConfig:
    return AgentConfig()

def get_tavily_client() -> Optional[Any]:
    global _client
    
    # Load config once or per call (lightweight)
    try:
        config = _get_config()
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
        config = _get_config()
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
        return format_error(ErrorType.CONFIG, "Search is unavailable due to missing configuration (API Key or SDK).")

    query = (query or "").strip()
    if not query:
        return format_error(ErrorType.VALIDATION, "Empty query provided.")

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
            return format_error(ErrorType.ACCESS_DENIED, "Invalid API credentials (401 Unauthorized).")
        return format_error(ErrorType.NETWORK, f"Search failed. Details: {str(e)}")

    results = []
    
    # AI Answer
    answer = response.get("answer")
    if answer:
        results.append(f"AI Overview:\n{answer}\n{'='*40}")

    items = response.get("results", [])
    if not items:
        return "\n".join(results) if results else format_error(ErrorType.NOT_FOUND, "No results found.")

    # Format compact result
    total_chars = 0
    max_chars = _SAFETY_POLICY.max_search_chars if _SAFETY_POLICY else 10000
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
    
    Args:
        urls: Single URL string or list of URL strings.
        advanced: If True, uses deeper extraction (slower).
    """
    client = get_tavily_client()
    if not client:
        return format_error(ErrorType.CONFIG, "Fetch unavailable due to missing configuration.")

    # Robust handling for models that pass stringified lists
    if isinstance(urls, str):
        urls = urls.strip()
        # Check if it looks like a JSON list
        if urls.startswith("[") and urls.endswith("]"):
            try:
                parsed = json.loads(urls)
                if isinstance(parsed, list):
                    urls = parsed
                else:
                    urls = [urls] # Fallback
            except json.JSONDecodeError:
                urls = [urls] # Treat as single URL if parse fails
        else:
            urls = [urls]
    
    clean_urls = []
    for u in urls:
        if isinstance(u, str):
             # Remove quotes and whitespace that models sometimes add
             u_clean = u.strip().strip('"\'')
             if u_clean.startswith("http"):
                 clean_urls.append(u_clean)
    
    clean_urls = list(set(clean_urls))[:20]
    if not clean_urls:
        return format_error(ErrorType.VALIDATION, f"No valid URLs provided. Input was: {urls}")

    depth = "advanced" if advanced else "basic"
    try:
        response = await _execute_with_retry(client.extract, urls=clean_urls, extract_depth=depth)
    except Exception as e:
        return format_error(ErrorType.EXECUTION, f"Fetch failed. Details: {e}")

    output_parts = []
    # Dynamic limit per URL based on count
    max_tool_output = _SAFETY_POLICY.max_tool_output if _SAFETY_POLICY else 4000
    max_chars_per_url = max_tool_output if len(clean_urls) == 1 else int(max_tool_output / len(clean_urls)) + 500
    
    for item in response.get("results", []):
        url = item.get("url", "Unknown")
        content = item.get("raw_content") or item.get("content") or ""
        
        content = truncate_output(content, max_chars_per_url, source=url)
        
        output_parts.append(f"=== SOURCE: {url} ===\n{content or '[Empty]'}\n{'='*30}")

    failed = response.get("failed_results", [])
    if failed:
        for f in failed:
            output_parts.append(f"❌ FAILED: {f.get('url')} - {f.get('error')}")

    return "\n".join(output_parts) or format_error(ErrorType.EXECUTION, "No content extracted.")

@tool("batch_web_search")
@with_cache(ttl=600)
async def batch_web_search(queries: List[str]) -> str:
    """
    Perform multiple searches in parallel.
    Args:
        queries: List of search queries
    """
    if not queries:
        return format_error(ErrorType.VALIDATION, "No queries provided.")
    
    results = []
    for q in queries[:5]: # Limit to 5 parallel searches
        res = await web_search(q)
        results.append(f"Query: {q}\n{res}\n{'='*50}")
        
    return "\n".join(results)
