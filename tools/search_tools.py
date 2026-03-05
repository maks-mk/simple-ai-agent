import ast
import asyncio
import json
import logging
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from langchain_core.tools import tool

from core.config import AgentConfig
from core.errors import ErrorType, format_error
from core.safety_policy import SafetyPolicy
from core.utils import truncate_output

logger = logging.getLogger(__name__)

# --- Tavily SDK init ---
try:
    from tavily import AsyncTavilyClient
except ImportError:
    AsyncTavilyClient = None
    logger.warning("Tavily SDK not installed. Search tools will be disabled.")

try:
    from tavily import errors as tavily_errors
except ImportError:
    tavily_errors = None

_client: Optional[Any] = None
_client_initialized: bool = False
_search_semaphore: Optional[asyncio.Semaphore] = None

# Global safety policy
_SAFETY_POLICY: Optional[SafetyPolicy] = None

# Runtime config injected from ToolRegistry
_RUNTIME_CONFIG: Optional[AgentConfig] = None

# In-memory cache: key -> (result, monotonic_ts)
_SEARCH_CACHE: Dict[str, tuple[Any, float]] = {}
_MAX_CACHE_SIZE = 50

# Tavily documented limits
_TAVILY_MAX_RESULTS = 20
_TAVILY_MAX_EXTRACT_URLS = 20
_MAX_BATCH_QUERIES = 5

_ALLOWED_SEARCH_DEPTHS = {"basic", "advanced", "fast", "ultra-fast"}
_ALLOWED_TOPICS = {"general", "news", "finance"}
_ALLOWED_FORMATS = {"markdown", "text"}


def _get_semaphore() -> asyncio.Semaphore:
    global _search_semaphore
    if _search_semaphore is None:
        _search_semaphore = asyncio.Semaphore(5)
    return _search_semaphore


def set_safety_policy(policy: SafetyPolicy):
    global _SAFETY_POLICY
    _SAFETY_POLICY = policy


def set_runtime_config(config: AgentConfig):
    global _RUNTIME_CONFIG
    _RUNTIME_CONFIG = config


# ======================================================
# Caching
# ======================================================

def _normalize_for_cache(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize_for_cache(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_for_cache(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _cache_key(func_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    payload = {
        "func": func_name,
        "args": _normalize_for_cache(args),
        "kwargs": _normalize_for_cache(kwargs),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _cleanup_cache(ttl: int):
    if not _SEARCH_CACHE:
        return

    now = time.monotonic()

    # Drop expired entries first
    expired = [k for k, (_, ts) in _SEARCH_CACHE.items() if now - ts >= ttl]
    for key in expired:
        _SEARCH_CACHE.pop(key, None)

    # Then enforce size limit
    if len(_SEARCH_CACHE) < _MAX_CACHE_SIZE:
        return

    items_by_age = sorted(_SEARCH_CACHE.items(), key=lambda item: item[1][1])
    # Keep a little headroom to reduce frequent cleanup churn
    target_size = int(_MAX_CACHE_SIZE * 0.85)
    remove_count = max(1, len(_SEARCH_CACHE) - target_size)
    for key, _ in items_by_age[:remove_count]:
        _SEARCH_CACHE.pop(key, None)


def with_cache(ttl: int = 600):
    """Async TTL cache decorator."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = _cache_key(func.__name__, args, kwargs)
            now = time.monotonic()

            cached = _SEARCH_CACHE.get(key)
            if cached:
                result, ts = cached
                if now - ts < ttl:
                    logger.debug("Cache hit for %s", func.__name__)
                    return result
                _SEARCH_CACHE.pop(key, None)

            result = await func(*args, **kwargs)

            # Cache only successful outputs
            if isinstance(result, str) and not result.lower().startswith(("error:", "ошибка:", "error[")):
                if len(_SEARCH_CACHE) >= _MAX_CACHE_SIZE:
                    _cleanup_cache(ttl=ttl)
                _SEARCH_CACHE[key] = (result, now)

            return result

        return wrapper

    return decorator


# ======================================================
# Helpers
# ======================================================

def _get_config() -> AgentConfig:
    if _RUNTIME_CONFIG is not None:
        return _RUNTIME_CONFIG
    return AgentConfig()


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value_int))


def _normalize_query(query: str) -> str:
    query = (query or "").strip()
    return " ".join(query.split())


def _is_valid_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _parse_urls_input(urls: Union[str, List[str]]) -> List[str]:
    candidates: List[str] = []

    if isinstance(urls, str):
        raw = urls.strip()
        parsed_obj: Any = None

        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed_obj = json.loads(raw)
            except json.JSONDecodeError:
                try:
                    parsed_obj = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    parsed_obj = None

        if isinstance(parsed_obj, list):
            candidates = [str(x) for x in parsed_obj]
        elif "," in raw:
            candidates = [part.strip() for part in raw.split(",")]
        else:
            candidates = [raw]
    elif isinstance(urls, list):
        candidates = [str(x) for x in urls]

    cleaned: List[str] = []
    seen = set()
    for item in candidates:
        u = item.strip().strip('"\'')
        if not u or not _is_valid_http_url(u) or u in seen:
            continue
        seen.add(u)
        cleaned.append(u)
        if len(cleaned) >= _TAVILY_MAX_EXTRACT_URLS:
            break

    return cleaned


def _format_tavily_error(exc: Exception) -> str:
    msg = str(exc)
    lower = msg.lower()

    if tavily_errors is not None:
        if isinstance(exc, getattr(tavily_errors, "MissingAPIKeyError", tuple())):
            return format_error(ErrorType.CONFIG, "TAVILY_API_KEY is missing.")
        if isinstance(exc, getattr(tavily_errors, "InvalidAPIKeyError", tuple())):
            return format_error(ErrorType.ACCESS_DENIED, "Invalid Tavily API key.")
        if isinstance(exc, getattr(tavily_errors, "UsageLimitExceededError", tuple())):
            return format_error(ErrorType.LIMIT_EXCEEDED, "Tavily usage limit exceeded.")
        if isinstance(exc, getattr(tavily_errors, "BadRequestError", tuple())):
            return format_error(ErrorType.VALIDATION, f"Bad Tavily request: {msg}")
        if isinstance(exc, getattr(tavily_errors, "ForbiddenError", tuple())):
            return format_error(ErrorType.ACCESS_DENIED, f"Tavily access forbidden: {msg}")
        if isinstance(exc, getattr(tavily_errors, "TimeoutError", tuple())):
            return format_error(ErrorType.TIMEOUT, f"Tavily request timed out: {msg}")

    if "401" in lower or "unauthorized" in lower or "invalid api key" in lower:
        return format_error(ErrorType.ACCESS_DENIED, "Invalid API credentials (401 Unauthorized).")
    if "timeout" in lower:
        return format_error(ErrorType.TIMEOUT, f"Request timed out: {msg}")
    if "limit" in lower or "quota" in lower:
        return format_error(ErrorType.LIMIT_EXCEEDED, f"Usage limit reached: {msg}")

    return format_error(ErrorType.NETWORK, f"Tavily request failed: {msg}")


def get_tavily_client() -> Optional[Any]:
    global _client, _client_initialized

    if _client_initialized:
        return _client

    _client_initialized = True

    try:
        config = _get_config()
    except Exception as e:
        logger.error("Config load failed: %s", e)
        return None

    if not config.enable_search_tools or AsyncTavilyClient is None:
        return None

    if not config.tavily_api_key:
        logger.warning("TAVILY_API_KEY is not set. Web search tools will return errors.")
        return None

    try:
        _client = AsyncTavilyClient(api_key=config.tavily_api_key.get_secret_value())
        return _client
    except Exception as e:
        logger.error("Failed to initialize Tavily client: %s", e)
        return None


async def _execute_with_retry(coroutine_func, *args, **kwargs):
    """Execute async request with retries and semaphore-based concurrency control."""
    try:
        config = _get_config()
        max_retries = _clamp_int(config.max_retries, low=1, high=8, default=3)
        retry_delay = max(0.1, float(config.retry_delay))
    except Exception:
        max_retries = 3
        retry_delay = 2.0

    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            async with _get_semaphore():
                return await coroutine_func(*args, **kwargs)
        except Exception as e:  # noqa: PERF203
            last_error = e
            if attempt < max_retries - 1:
                # Simple exponential backoff, capped to keep UX responsive
                backoff = min(retry_delay * (2 ** attempt), 10.0)
                await asyncio.sleep(backoff)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Unknown retry failure")


# ======================================================
# Tools
# ======================================================

@with_cache(ttl=600)
async def _web_search_impl(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    topic: str = "general",
) -> str:
    """Implementation of web search logic."""
    client = get_tavily_client()
    if not client:
        return format_error(
            ErrorType.CONFIG,
            "Search is unavailable due to missing configuration (TAVILY_API_KEY or SDK).",
        )

    query = _normalize_query(query)
    if not query:
        return format_error(ErrorType.VALIDATION, "Empty query provided.")

    max_results = _clamp_int(max_results, low=1, high=_TAVILY_MAX_RESULTS, default=5)

    search_depth = (search_depth or "basic").strip().lower()
    if search_depth not in _ALLOWED_SEARCH_DEPTHS:
        return format_error(
            ErrorType.VALIDATION,
            f"Invalid search_depth '{search_depth}'. Allowed: {sorted(_ALLOWED_SEARCH_DEPTHS)}",
        )

    topic = (topic or "general").strip().lower()
    if topic not in _ALLOWED_TOPICS:
        return format_error(
            ErrorType.VALIDATION,
            f"Invalid topic '{topic}'. Allowed: {sorted(_ALLOWED_TOPICS)}",
        )

    try:
        response = await _execute_with_retry(
            client.search,
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            topic=topic,
            include_answer=True,
            include_usage=True,
        )
    except Exception as e:
        return _format_tavily_error(e)

    parts: List[str] = []

    answer = response.get("answer")
    if answer:
        parts.append(f"AI Overview:\n{answer}\n{'=' * 40}")

    items = response.get("results", [])
    if not items:
        return "\n".join(parts) if parts else format_error(ErrorType.NOT_FOUND, "No results found.")

    max_chars = _SAFETY_POLICY.max_search_chars if _SAFETY_POLICY else 10000
    total_chars = 0
    separator = "-" * 40

    for item in items:
        title = item.get("title") or "Untitled"
        url = item.get("url") or ""
        score = item.get("score", 0)
        content = item.get("content") or ""

        if not content:
            continue

        block = f"Source: {title} (score: {float(score):.2f})\nURL: {url}\n{content}"
        if total_chars + len(block) > max_chars:
            break

        parts.append(block)
        parts.append(separator)
        total_chars += len(block) + len(separator)

    usage = response.get("usage")
    if usage:
        parts.append(f"Usage: {usage}")

    parts.append("[Search completed. Use the context above.]")
    return "\n".join(parts)


@tool("web_search")
async def web_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    topic: str = "general",
) -> str:
    """
    Search internet for snippets and AI summary.

    Args:
        query: Search query.
        max_results: Number of results (1..20).
        search_depth: Tavily search depth: basic, advanced, fast, ultra-fast.
        topic: general, news, finance.
    """
    return await _web_search_impl(query, max_results, search_depth, topic)


@tool("fetch_content")
@with_cache(ttl=1800)
async def fetch_content(
    urls: Union[str, List[str]],
    advanced: bool = False,
    content_format: str = "markdown",
    query: Optional[str] = None,
    chunks_per_source: int = 3,
) -> str:
    """
    Extract text from one or multiple URLs.

    Args:
        urls: Single URL or list of URLs (up to 20).
        advanced: Use advanced extraction depth.
        content_format: markdown or text.
        query: Optional query for extracting relevant chunks.
        chunks_per_source: Relevant chunks per source when query is set.
    """
    client = get_tavily_client()
    if not client:
        return format_error(ErrorType.CONFIG, "Fetch unavailable due to missing configuration.")

    clean_urls = _parse_urls_input(urls)
    if not clean_urls:
        return format_error(ErrorType.VALIDATION, "No valid HTTP/HTTPS URLs provided.")

    extract_depth = "advanced" if advanced else "basic"

    content_format = (content_format or "markdown").strip().lower()
    if content_format not in _ALLOWED_FORMATS:
        return format_error(
            ErrorType.VALIDATION,
            f"Invalid content_format '{content_format}'. Allowed: {sorted(_ALLOWED_FORMATS)}",
        )

    request_kwargs: Dict[str, Any] = {
        "urls": clean_urls,
        "extract_depth": extract_depth,
        "format": content_format,
    }

    if query:
        request_kwargs["query"] = _normalize_query(query)
        request_kwargs["chunks_per_source"] = _clamp_int(chunks_per_source, 1, 20, 3)

    try:
        response = await _execute_with_retry(client.extract, **request_kwargs)
    except Exception as e:
        return _format_tavily_error(e)

    output_parts: List[str] = []
    max_chars_limit = _SAFETY_POLICY.max_search_chars if _SAFETY_POLICY else 15000
    max_chars_per_url = max(1000, int(max_chars_limit / max(1, len(clean_urls))))

    for item in response.get("results", []):
        url = item.get("url", "Unknown")
        content = item.get("raw_content") or item.get("content") or ""
        content = truncate_output(content, max_chars_per_url, source=url)
        output_parts.append(f"=== SOURCE: {url} ===\n{content or '[Empty]'}\n{'=' * 30}")

    failed = response.get("failed_results", [])
    if failed:
        for failed_item in failed:
            output_parts.append(
                f"FAILED: {failed_item.get('url', 'unknown')} - {failed_item.get('error', 'Unknown error')}"
            )

    usage = response.get("usage")
    if usage:
        output_parts.append(f"Usage: {usage}")

    return "\n".join(output_parts) or format_error(ErrorType.EXECUTION, "No content extracted.")


@tool("batch_web_search")
@with_cache(ttl=600)
async def batch_web_search(
    queries: List[str],
    max_results: int = 5,
    search_depth: str = "basic",
    topic: str = "general",
) -> str:
    """
    Perform multiple searches in parallel.

    Args:
        queries: List of search queries.
        max_results: Results per query.
        search_depth: Tavily search depth.
        topic: general, news, finance.
    """
    if not queries:
        return format_error(ErrorType.VALIDATION, "No queries provided.")

    normalized_queries: List[str] = []
    seen = set()
    for query in queries:
        q = _normalize_query(str(query))
        if not q or q in seen:
            continue
        seen.add(q)
        normalized_queries.append(q)

    if not normalized_queries:
        return format_error(ErrorType.VALIDATION, "No valid queries provided.")

    selected = normalized_queries[:_MAX_BATCH_QUERIES]
    output: List[str] = []

    if len(normalized_queries) > _MAX_BATCH_QUERIES:
        output.append(
            f"WARNING: Only first {_MAX_BATCH_QUERIES} unique queries were executed to prevent API limits.\n"
            + "=" * 40
        )

    tasks = [_web_search_impl(q, max_results=max_results, search_depth=search_depth, topic=topic) for q in selected]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for query, result in zip(selected, results):
        if isinstance(result, Exception):
            rendered = format_error(ErrorType.EXECUTION, f"Unhandled batch search error: {result}")
        else:
            rendered = str(result)
        output.append(f"Query: {query}\n{rendered}\n{'=' * 50}")

    return "\n".join(output)

