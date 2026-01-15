import os
import logging
import asyncio
from typing import Optional, List, Any, Union

logger = logging.getLogger(__name__)

# --- Инициализация Tavily ---
try:
    from tavily import AsyncTavilyClient
except ImportError:
    AsyncTavilyClient = None
    logger.warning("Tavily SDK not installed. Search tools will be disabled.")

_client: Optional[Any] = None

def get_tavily_client() -> Optional[Any]:
    global _client
    if _client is not None:
        return _client

    if AsyncTavilyClient is None:
        logger.error("Tavily SDK is not available (ImportError).")
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


# ----------------------------
# Tool 1: Web Search (Широкий поиск)
# ----------------------------

async def web_search(query: str, max_results: int = 5) -> str:
    """
    Search internet for information. Returns snippets from multiple sources + AI summary.
    Use for: quick facts, news, comparing sources. Don't use for full article text.
    """
    client = get_tavily_client()
    if not client:
        return "System: Search is unavailable due to missing configuration."

    query = (query or "").strip()
    if not query:
        return "System: Search completed: empty query provided."

    response = None
    last_error = None

    # --- RETRY LOGIC ---
    for attempt in range(3):
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
            logger.warning(f"Web search attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                await asyncio.sleep(2)

    if response is None:
        msg = str(last_error).lower()
        if "401" in msg or "unauthorized" in msg:
            return "System: Search failed due to invalid API credentials."
        return f"System: Search failed after 3 attempts. Error: {msg}"

    results: List[str] = []

    # AI-generated answer
    answer = response.get("answer")
    if answer:
        results.append(f"AI Overview:\n{answer}")
        results.append("=" * 40)

    items = response.get("results", [])
    if not items:
        if results:
            return "\n".join(results)
        return "System: Search completed: no results found."

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
        return "System: Fetch unavailable."

    # 1. Нормализация входных данных
    if isinstance(urls, str):
        # Если пришла строка, делаем из нее список
        urls = [urls]
    
    if not isinstance(urls, list) or not urls:
        return "System: Error - 'urls' must be a string or a list of strings."

    # 2. Чистка и валидация
    clean_urls = []
    for u in urls:
        if isinstance(u, str):
            u_clean = u.strip().strip('"').strip("'")
            if u_clean.startswith("http"):
                clean_urls.append(u_clean)
    
    clean_urls = list(set(clean_urls)) # Убираем дубликаты
    
    if not clean_urls:
        return "System: No valid URLs provided."

    if len(clean_urls) > 20:
        clean_urls = clean_urls[:20]
        logger.warning("Fetch batch truncated to first 20 URLs.")

    # 3. Настройка параметров
    depth = "advanced" if advanced else "basic"
    response = None
    last_error = None

    # 4. Retry Logic
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
        return f"System: Fetch failed. Error: {str(last_error)}"

    # 5. Формирование отчета
    output_parts = []
    # Если ссылка одна, даем больше лимит на символы, если много - меньше
    max_chars_per_url = 15000 if len(clean_urls) == 1 else 6000 

    results = response.get("results", [])
    
    # Сценарий одиночной ссылки (возвращаем чистый текст без лишних заголовков)
    if len(clean_urls) == 1 and len(results) == 1 and not response.get("failed_results"):
        data = results[0]
        content = data.get("raw_content") or data.get("content") or ""
        if not content: return "System: Empty content."
        
        if len(content) > max_chars_per_url:
            content = content[:max_chars_per_url] + f"\n... [Truncated]"
        return f"URL: {clean_urls[0]} (Mode: {depth})\n\n{content}\n\n[Content extracted]"

    # Сценарий списка ссылок (Batch)
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
        return "System: No content extracted."

    return "\n".join(output_parts)


# ----------------------------
# Tool 3: Deep Search (Опционально)
# ----------------------------

async def deep_search(query: str, max_results: int = 3) -> str:
    """
    Deep search with full page content. Returns complete text from top results.
    WARNING: Uses many tokens! Use only for comprehensive research/reports.
    """
    client = get_tavily_client()
    if not client:
        return "System: Deep search is unavailable due to missing configuration."

    query = (query or "").strip()
    if not query:
        return "System: Deep search completed: empty query provided."

    response = None
    last_error = None

    # --- RETRY LOGIC ---
    for attempt in range(3):
        try:
            response = await client.search(
                query=query,
                max_results=max_results,  # Fewer results due to large volume
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True,  # FIXED: Boolean True instead of "text"
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
            return "System: Deep search failed due to invalid API credentials."
        return f"System: Deep search failed after 3 attempts. Error: {msg}"

    results: List[str] = []

    # AI overview
    answer = response.get("answer")
    if answer:
        results.append(f"AI Overview:\n{answer}")
        results.append("=" * 40)

    items = response.get("results", [])
    if not items:
        if results:
            return "\n".join(results)
        return "System: Deep search completed: no results found."

    # Format detailed result with full content
    total_chars = 0
    max_chars = 30000  # Higher limit for deep search
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
# Tool Registration Metadata
# ----------------------------

web_search.name = "web_search"
web_search.description = "Search internet. Returns snippets + AI summary. Use for quick facts/news."

fetch_content.name = "fetch_content"
fetch_content.description = "Extract text from URL(s). Can handle a single link or a list. Args: urls=['link1', 'link2']"

deep_search.name = "deep_search"
deep_search.description = "Deep search with full content. High token cost. Use for research only."