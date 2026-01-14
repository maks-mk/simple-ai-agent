import logging
import hashlib
import asyncio
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

# Опциональные импорты
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    chromadb = None
    SentenceTransformer = None

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

        if chromadb is None or SentenceTransformer is None:
            raise ImportError("Install deps: pip install chromadb sentence-transformers")

        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.top_k = top_k

        try:
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(name="memory")
            logger.info(f"📂 Память подключена: {db_path}")
        except Exception as e:
            logger.critical(f"❌ Ошибка ChromaDB: {e}")
            raise e

    @property
    def model(self):
        """Ленивая загрузка модели: грузится только при первом использовании."""
        if MemoryManager._model_instance is None:
            logger.info(f"⏳ Загрузка модели: {self.embedding_model_name}...")
            MemoryManager._model_instance = SentenceTransformer(self.embedding_model_name)
        return MemoryManager._model_instance

    @staticmethod
    @lru_cache(maxsize=256)
    def _generate_id(text: str) -> str:
        """Детерминированная генерация ID на основе текста."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def _get_embedding(self, text: str) -> List[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    # ================= СИНХРОННЫЕ МЕТОДЫ =================

    def remember(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        if not text.strip(): return "Empty text"
        try:
            doc_id = self._generate_id(text)
            emb = self._get_embedding(text)
            self.collection.upsert(
                ids=[doc_id],
                documents=[text],
                embeddings=[emb],
                metadatas=[metadata or {"source": "user"}]
            )
            logger.info(f"Memory Saved: {text[:30]}...")
            return f"Запомнил: {text[:50]}..."
        except Exception as e:
            logger.error(f"Memory Error: {e}")
            return f"Ошибка записи: {e}"

    def recall(self, query: str, n_results: Optional[int] = None) -> List[str]:
        if not query.strip() or self.collection.count() == 0:
            return []
        try:
            emb = self._get_embedding(query)
            results = self.collection.query(
                query_embeddings=[emb],
                n_results=n_results or self.top_k
            )
            return results.get("documents", [[]])[0]
        except Exception as e:
            logger.error(f"Recall Error: {e}")
            return []

    def delete_fact_by_query(self, query: str, n_results: int = 1) -> int:
        if not query.strip() or self.collection.count() == 0:
            return 0
            
        try:
            emb = self._get_embedding(query)
            results = self.collection.query(
                query_embeddings=[emb],
                n_results=n_results
            )
            
            ids_to_delete = results.get("ids", [[]])[0]
            docs_to_delete = results.get("documents", [[]])[0]
            
            if not ids_to_delete:
                return 0

            self.collection.delete(ids=ids_to_delete)
            logger.warning(f"🗑️ Удалено: {docs_to_delete}")
            return len(ids_to_delete)
            
        except Exception as e:
            logger.error(f"Delete Error: {e}")
            return 0

    def wipe_memory(self) -> str:
        try:
            self.client.delete_collection("memory")
            self.collection = self.client.get_or_create_collection("memory")
            return "Память полностью очищена."
        except Exception as e:
            return f"Ошибка очистки: {e}"

    # ================= АСИНХРОННЫЕ ОБЕРТКИ =================

    async def aremember(self, text: str, metadata: dict = None) -> str:
        return await asyncio.to_thread(self.remember, text, metadata)

    async def arecall(self, query: str) -> List[str]:
        return await asyncio.to_thread(self.recall, query)

    async def adelete_fact_by_query(self, query: str) -> int:
        return await asyncio.to_thread(self.delete_fact_by_query, query)