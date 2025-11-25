"""
memory_manager.py

Модуль для долговременной памяти AI-агента с использованием ChromaDB.
Оптимизирован: deque для сессии, нормализация эмбеддингов, Singleton для модели.
"""

import logging
import hashlib
import functools
from typing import List, Dict, Optional, Any
from collections import deque

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    chromadb = None  # type: ignore
    SentenceTransformer = None  # type: ignore

logger = logging.getLogger(__name__)

class MemoryManager:
    _model_instance = None  # Singleton для модели класса

    def __init__(
        self,
        db_path: str = "./memory_db",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 5,
        session_size: int = 6
    ):
        if chromadb is None or SentenceTransformer is None:
            raise ImportError("Не установлены зависимости: pip install chromadb sentence-transformers")

        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        
        # ОПТИМИЗАЦИЯ: deque автоматически удаляет старые элементы при переполнении
        self.session_history: deque = deque(maxlen=session_size)

        # 1. Инициализация ChromaDB
        try:
            # settings=chromadb.Settings(anonymized_telemetry=False) можно добавить для отключения телеметрии
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(name="memory")
            logger.info(f"📂 Память подключена: {db_path}")
        except Exception as e:
            logger.critical(f"❌ Критическая ошибка ChromaDB: {e}")
            raise e

        # 2. Ленивая загрузка модели
        self._load_model()

    def _load_model(self):
        """Singleton загрузка тяжелой модели трансформеров."""
        if MemoryManager._model_instance is None:
            logger.info(f"⏳ Загрузка модели эмбеддингов: {self.embedding_model_name}...")
            # ОПТИМИЗАЦИЯ: device='cpu' явно, если нужно, или auto
            MemoryManager._model_instance = SentenceTransformer(self.embedding_model_name)
            logger.info("✅ Модель загружена.")
        self.model = MemoryManager._model_instance

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def _generate_id(text: str) -> str:
        """Генерирует ID (SHA256). Кэшируется для повторяющихся строк."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def _get_embedding(self, text: str) -> List[float]:
        """Генерирует нормализованный эмбеддинг."""
        # ОПТИМИЗАЦИЯ: normalize_embeddings=True улучшает поиск через cosine similarity
        return self.model.encode([text], show_progress_bar=False, normalize_embeddings=True)[0].tolist()

    # ------------------ Долговременная память ------------------

    def remember(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Сохраняет факт (Upsert)."""
        if not text or not text.strip():
            return False

        text = text.strip()
        doc_id = self._generate_id(text)
        
        try:
            emb = self._get_embedding(text)
            self.collection.upsert(
                ids=[doc_id],
                documents=[text],
                embeddings=[emb],
                metadatas=[metadata or {"source": "user", "type": "general"}]
            )
            logger.info(f"💾 Запомнил: {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения: {e}")
            return False

    def recall(self, query: str, n_results: int = None) -> List[str]:
        """Поиск похожих фактов."""
        if not query.strip() or self.collection.count() == 0:
            return []

        limit = n_results or self.top_k
        try:
            emb = self._get_embedding(query)
            results = self.collection.query(
                query_embeddings=[emb],
                n_results=limit
            )
            # results['documents'] это [[doc1, doc2, ...]]
            return results.get("documents", [[]])[0]
        except Exception as e:
            logger.error(f"❌ Ошибка поиска: {e}")
            return []

    def delete_fact_by_query(self, query: str, n_results: int = 1) -> int:
        """Находит факт по смыслу и удаляет его."""
        if not query.strip() or self.collection.count() == 0:
            return 0
            
        try:
            # 1. Находим кандидатов
            candidates = self.recall(query, n_results=n_results)
            if not candidates:
                return 0

            # 2. Вычисляем их ID
            ids_to_delete = [self._generate_id(text) for text in candidates]
            
            # 3. Удаляем
            self.collection.delete(ids=ids_to_delete)
            logger.warning(f"🗑️ Удалено фактов: {len(ids_to_delete)} (запрос: '{query}')")
            return len(ids_to_delete)
            
        except Exception as e:
            logger.error(f"❌ Ошибка удаления: {e}")
            return 0

    def wipe_memory(self):
        """Полная очистка."""
        try:
            self.client.delete_collection("memory")
            self.collection = self.client.get_or_create_collection("memory")
            logger.warning("🧹 Память очищена.")
        except Exception as e:
            logger.error(f"Ошибка вайпа: {e}")

    # ------------------ Сессия (Deque) ------------------

    def add_to_session(self, role: str, content: str):
        if content:
            self.session_history.append({"role": role, "content": content})

    def get_session_history(self) -> List[Dict[str, str]]:
        return list(self.session_history)
    
    def clear_session(self):
        self.session_history.clear()