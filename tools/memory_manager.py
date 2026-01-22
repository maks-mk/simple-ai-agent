import logging
import hashlib
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any
from functools import lru_cache
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Опциональные импорты
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    DEPS_INSTALLED = True
except ImportError:
    chromadb = None
    SentenceTransformer = None
    DEPS_INSTALLED = False

class MemoryManager:
    _instance = None
    _model_instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MemoryManager, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        db_path: str = "./memory_db",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 5
    ):
        if hasattr(self, 'client'):
            return

        if not DEPS_INSTALLED:
            logger.warning("Memory dependencies missing (chromadb, sentence-transformers). Memory disabled.")
            return

        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.top_k = top_k

        try:
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(name="memory")
            logger.info(f"📂 Memory connected: {db_path}")
        except Exception as e:
            logger.error(f"❌ ChromaDB Error: {e}")
            self.client = None

    @property
    def model(self):
        if not DEPS_INSTALLED: return None
        if MemoryManager._model_instance is None:
            logger.info(f"⏳ Loading embedding model: {self.embedding_model_name}...")
            MemoryManager._model_instance = SentenceTransformer(self.embedding_model_name)
        return MemoryManager._model_instance

    @staticmethod
    @lru_cache(maxsize=256)
    def _generate_id(text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def _get_embedding(self, text: str) -> List[float]:
        if not self.model: return []
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    # --- Core Logic ---

    def remember(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        if not self.client: return "Memory system unavailable."
        if not text.strip(): return "Empty text."
        try:
            doc_id = self._generate_id(text)
            emb = self._get_embedding(text)
            self.collection.upsert(
                ids=[doc_id],
                documents=[text],
                embeddings=[emb],
                metadatas=[metadata or {"source": "user"}]
            )
            return f"Saved to memory: {text[:50]}..."
        except Exception as e:
            return f"Memory write error: {e}"

    def recall(self, query: str, n_results: Optional[int] = None) -> List[str]:
        if not self.client or self.collection.count() == 0: return []
        try:
            emb = self._get_embedding(query)
            results = self.collection.query(
                query_embeddings=[emb],
                n_results=n_results or self.top_k
            )
            return results.get("documents", [[]])[0]
        except Exception as e:
            logger.error(f"Recall error: {e}")
            return []

    def delete_fact(self, query: str) -> str:
        if not self.client: return "Memory unavailable."
        try:
            emb = self._get_embedding(query)
            results = self.collection.query(query_embeddings=[emb], n_results=1)
            ids = results.get("ids", [[]])[0]
            if ids:
                self.collection.delete(ids=ids)
                return "Fact deleted."
            return "Fact not found."
        except Exception as e:
            return f"Delete error: {e}"

    # --- Async Wrappers ---
    async def aremember(self, text: str, metadata: dict = None) -> str:
        return await asyncio.to_thread(self.remember, text, metadata)

    async def arecall(self, query: str) -> List[str]:
        return await asyncio.to_thread(self.recall, query)
        
    async def adelete(self, query: str) -> str:
        return await asyncio.to_thread(self.delete_fact, query)

# ==========================================
# EXPORTED TOOLS (Native @tool decorators)
# ==========================================

# Инициализируем синглтон (параметры возьмутся дефолтные или можно настроить через конфиг позже)
_memory = MemoryManager()

@tool("remember_fact")
async def remember_fact(text: str, category: str = "general") -> str:
    """
    Saves important information to long-term memory. 
    Use this for user preferences, facts about projects, or specific instructions.
    """
    return await _memory.aremember(text, {"type": category})

@tool("recall_facts")
async def recall_facts(query: str) -> str:
    """
    Searches long-term memory for relevant facts.
    Args:
        query: The topic or question to search for in memory.
    """
    facts = await _memory.arecall(query)
    return "\n".join(f"- {f}" for f in facts) if facts else "No relevant facts found in memory."

@tool("forget_fact")
async def forget_fact(query: str) -> str:
    """
    Removes a fact from memory that matches the query.
    """
    return await _memory.adelete(query)