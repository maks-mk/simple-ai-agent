import os
import logging
import asyncio
import json
import hashlib
import time
from functools import wraps
from typing import Optional, List, Any, Union, Dict
from langchain_core.tools import tool
# from core.config import AgentConfig  <-- Removed to avoid tight coupling


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

# Глобальный словарь для кэша: { "hash_key": (result_str, timestamp_float) }
_SEARCH_CACHE = {}
_MAX_CACHE_SIZE = 50  # Ограничение количества записей

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
    Ключ кэша формируется из аргументов функции.
    Не кэширует ответы, содержащие ошибки.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 1. Генерация уникального ключа
            try:
                # Сериализуем аргументы в JSON для хэширования
                key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
                key = hashlib.md5(key_data.encode()).hexdigest()
            except Exception:
                # Если аргументы сложные (не сериализуемые), пропускаем кэш
                return await func(*args, **kwargs)

            # 2. Проверка кэша (Hit)
            if key in _SEARCH_CACHE:
                result, timestamp = _SEARCH_CACHE[key]
                if time.time() - timestamp < ttl:
                    logger.info(f"⚡ Cache hit for {func.__name__} (key: {key[:8]})")
                    return result
                else:
                    # Удаляем протухший (Expired)
                    del _SEARCH_CACHE[key]

            # 3. Выполнение функции (Miss)
            result = await func(*args, **kwargs)

            # 4. Сохранение (только если это не ошибка)
            # Мы проверяем начало строки на типичные маркеры ошибок, чтобы не запоминать сбои
            if isinstance(result, str) and not result.lower().startswith(("error:", "ошибка:")):
                # Очистка перед вставкой
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
        "invalid_input": "Error: Invalid input. {details}",
        "fetch_error": "Error: Fetch failed. Details: {details}",
    }
    msg = error_templates.get(error_type, f"Error: {details}")
    if "{details}" in msg:
        msg = msg.format(details=details)
    return msg

def get_tavily_client() -> Optional[Any]:
    global _client
    
    # Проверка флага конфигурации (через env, чтобы не грузить весь Config)
    if os.getenv("ENABLE_SEARCH_TOOLS", "True").lower() == "false":
        return None 

    if _client is not None:
        return _client

    if AsyncTavilyClient is None:
        return None

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY is not set.")
        return None

    try:
        _client = AsyncTavilyClient(api_key=api_key)
        return _client
    except Exception as e:
        logger.error(f"Failed to initialize Tavily client: {e}")
        return None


# ======================================================
# TOOLS
# ======================================================

# ----------------------------
# Tool 1: Web Search (Широкий поиск)
# ----------------------------

@tool("web_search")
@with_cache(ttl=600)  # Кэш 10 минут
async def web_search(query: str, max_results: int = 5) -> str:
    """
    Search internet for information. Returns snippets from multiple sources + AI summary.
    Use for: quick facts, news, comparing sources. Don't use for full article text.
    """
    try:
        if os.getenv("ENABLE_SEARCH_TOOLS", "True").lower() == "false":
            return _format_error("disabled")
    except: pass

    client = get_tavily_client()
    if not client:
        return _format_error("missing_config")

    query = (query or "").strip()
    if not query:
        return _format_error("empty_query")

    response = None
    last_error = None

    # --- RETRY LOGIC WITH SEMAPHORE ---
    # Читаем настройки ретраев напрямую из env или используем дефолты
    max_retries = int(os.getenv("MAX_RETRIES", "3"))
    retry_delay = int(os.getenv("RETRY_DELAY", "2"))

    async with _search_semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.search(
                    query=query,
                    max_results=max_results,
                    search_depth="basic",
                    include_answer=True,
                )
                break 
            except Exception as e:
                last_error = e
                logger.warning(f"Web search attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)

    if response is None:
        msg = str(last_error).lower()
        if "401" in msg or "unauthorized" in msg:
            return _format_error("auth_error")
        return _format_error("api_error", details=str(last_error))

    results = []

    # AI-generated answer
    answer = response.get("answer")
    if answer:
        results.append(f"AI Overview:\n{answer}")
        results.append("=" * 40)

    items = response.get("results", [])
    if not items:
        if results:
            return "\n".join(results)
        return "Error: No results found."

    # Format compact result
    total_chars = 0
    max_chars = 11000 
    separator = "-" * 40

    for item in items:
        title = item.get("title") or "Untitled"
        url = item.get("url") or ""
        content = item.get("content") or ""
        score = item.get("score", 0)

        if not content:
            continue

        header = f"Source: {title} (score: {score:.2f})\nURL: {url}"
        block_overhead = len(header) + len(separator) + 5
        
        available_space = max_chars - total_chars - block_overhead
        if available_space <= 50:
            break

        if len(content) > available_space:
            content = content[:available_space] + "... (truncated)"

        block = f"{header}\n{content}"
        results.append(block)
        results.append(separator)
        total_chars += len(block) + len(separator)

    results.append("[Search completed. Use the context above.]")
    return "\n".join(results)


# ----------------------------
# Tool 2: Fetch Content (Универсальное чтение)
# ----------------------------

@tool("fetch_content")
@with_cache(ttl=1800)  # Кэш 30 минут (контент страниц меняется редко)
async def fetch_content(urls: Union[str, List[str]], advanced: bool = False) -> str:
    """
    Extract text from one or multiple URLs. 
    Use this to read pages after searching. Supports batching (up to 20 links).
    
    Args:
        urls: A single URL string OR a list of URL strings.
        advanced: Set True for better extraction on difficult sites (costs more).
    """
    client = get_tavily_client()
    if not client:
        return "Error: Fetch unavailable due to missing configuration."

    # 1. Нормализация входных данных
    if isinstance(urls, str):
        urls = [urls]
    
    if not isinstance(urls, list) or not urls:
        return "Error: 'urls' must be a string or a list of strings."

    # 2. Чистка и валидация
    clean_urls = []
    for u in urls:
        if isinstance(u, str):
            u_clean = u.strip().strip('"').strip("'")
            if u_clean.startswith("http"):
                clean_urls.append(u_clean)
    
    clean_urls = list(set(clean_urls)) 
    
    if not clean_urls:
        return "Error: No valid URLs provided."

    if len(clean_urls) > 20:
        clean_urls = clean_urls[:20]
        logger.warning("Fetch batch truncated to first 20 URLs.")

    # 3. Выполнение запроса
    depth = "advanced" if advanced else "basic"
    response = None
    last_error = None

    for attempt in range(3):
        try:
            response = await client.extract(urls=clean_urls, extract_depth=depth)
            break
        except Exception as e:
            last_error = e
            logger.warning(f"Fetch attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                await asyncio.sleep(2)

    if response is None:
        return f"Error: Fetch failed. Details: {str(last_error)}"

    # 4. Формирование отчета
    output_parts = []
    max_chars_per_url = 15000 if len(clean_urls) == 1 else 6000 

    results = response.get("results", [])
    
    # Одиночная ссылка
    if len(clean_urls) == 1 and len(results) == 1 and not response.get("failed_results"):
        data = results[0]
        content = data.get("raw_content") or data.get("content") or ""
        if not content: return "Error: Empty content."
        
        if len(content) > max_chars_per_url:
            content = content[:max_chars_per_url] + f"\n... [Truncated]"
        return f"URL: {clean_urls[0]} (Mode: {depth})\n\n{content}\n\n[Content extracted]"

    # Batch (пакет)
    for item in results:
        url = item.get("url", "Unknown URL")
        content = item.get("raw_content") or item.get("content") or ""
        
        if len(content) > max_chars_per_url:
            content = content[:max_chars_per_url] + f"\n... [Truncated]"
        
        if not content:
            content = "[Empty content]"

        block = (
            f"=== SOURCE: {url} ===\n"
            f"{content}\n"
            f"=================================\n"
        )
        output_parts.append(block)

    failed = response.get("failed_results", [])
    if failed:
        for f in failed:
            output_parts.append(f"❌ FAILED: {f.get('url')} - {f.get('error')}")

    if not output_parts:
        return "Error: No content extracted."

    return "\n".join(output_parts)


# ----------------------------
# Tool 3: Deep Search (Глубокий поиск)
# ----------------------------

@tool("deep_search")
@with_cache(ttl=3600)  # Кэш 1 час (дорогой запрос)
async def deep_search(query: str, max_results: int = 3) -> str:
    """
    Deep search with full page content. Returns complete text from top results.
    WARNING: Uses many tokens! Use only for comprehensive research/reports.
    """
    client = get_tavily_client()
    if not client:
        return "Error: Deep search is unavailable due to missing configuration."

    query = (query or "").strip()
    if not query:
        return "Error: Empty query provided."

    response = None
    last_error = None

    for attempt in range(3):
        try:
            response = await client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True, 
            )
            break
        except Exception as e:
            last_error = e
            logger.warning(f"Deep search attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                await asyncio.sleep(2)

    if response is None:
        msg = str(last_error).lower()
        if "401" in msg or "unauthorized" in msg:
            return "Error: Deep search failed due to invalid API credentials."
        return f"Error: Deep search failed after 3 attempts. Details: {msg}"

    results: List[str] = []

    answer = response.get("answer")
    if answer:
        results.append(f"AI Overview:\n{answer}")
        results.append("=" * 40)

    items = response.get("results", [])
    if not items:
        if results:
            return "\n".join(results)
        return "Error: No results found."

    total_chars = 0
    max_chars = 30000 
    separator = "=" * 60

    for idx, item in enumerate(items, 1):
        title = item.get("title") or "Untitled"
        url = item.get("url") or ""
        content = item.get("raw_content") or item.get("content") or ""
        score = item.get("score", 0)

        if not content:
            continue

        header = f"[Source {idx}] {title} (relevance: {score:.2f})\nURL: {url}\n"
        block_overhead = len(header) + len(separator) + 10
        
        available_space = max_chars - total_chars - block_overhead
        if available_space <= 100:
            break

        if len(content) > available_space:
            content = content[:available_space] + "\n... [Content truncated]"

        block = f"{header}{separator}\n{content}\n"
        results.append(block)
        total_chars += len(block)

    results.append("[Deep search completed. Full content extracted.]")
    return "\n".join(results)


# ----------------------------
# Tool 4: Batch Web Search
# ----------------------------

@tool("batch_web_search")
@with_cache(ttl=600)
async def batch_web_search(queries: List[str]) -> str:
    """
    Perform multiple web searches in parallel. 
    Use this tool when the user asks a complex question that requires researching 
    multiple different aspects or topics simultaneously.
    """
    client = get_tavily_client()
    if not client:
        return "Error: Search unavailable due to missing configuration."

    if len(queries) > 5:
        queries = queries[:5]
        
    tasks = []
    for q in queries:
        tasks.append(
            client.search(
                query=q,
                max_results=4, 
                search_depth="basic",
                include_answer=True
            )
        )

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


# ----------------------------
# Tool 5: Crawl Site
# ----------------------------

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
    if not client:
        return "Error: Crawl is unavailable due to missing configuration."

    start_url = (start_url or "").strip()
    if not start_url.startswith("http"):
        return "Error: 'start_url' must be a valid URL."

    response = None
    last_error = None

    max_retries = int(os.getenv("MAX_RETRIES", "3"))
    retry_delay = int(os.getenv("RETRY_DELAY", "2"))

    async with _search_semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.crawl(
                    url=start_url,
                    max_depth=max_depth,
                    limit=limit,
                    include_subdomains=include_subdomains,
                )
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Crawl attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)

    if response is None:
        return f"Error: Crawl failed. Details: {str(last_error)}"

    results = response.get("results", []) or response.get("pages", []) or []

    if not results:
        return "Error: No pages crawled."

    parts = []
    max_chars = 30000
    total = 0

    for idx, page in enumerate(results, 1):
        url = page.get("url", "Unknown URL")
        content = page.get("raw_content") or page.get("content") or ""
        if not content:
            continue

        header = f"[Page {idx}] {url}\n" + "=" * 60 + "\n"
        block = header + content + "\n\n"
        if total + len(block) > max_chars:
            parts.append("... [Truncated crawl output]")
            break

        parts.append(block)
        total += len(block)

    if not parts:
        return "Error: No content extracted from crawled pages."

    parts.append("[Crawl completed.]")
    return "\n".join(parts)