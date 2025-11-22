"""
memory_manager.py

Модуль для долговременной памяти AI-агента с использованием ChromaDB и мультиязычных эмбеддингов.
Оптимизирован для производительности и исключения дубликатов.
"""

import logging
import hashlib
from typing import List, Dict, Optional, Any
import os

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Для работы памяти установите: pip install chromadb sentence-transformers")

# Настройка логгера
logger = logging.getLogger(__name__)

class MemoryManager:
    _model_instance = None  # Singleton для модели (чтобы не грузить её дважды)

    def __init__(
        self,
        db_path: str = "./memory_db",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 5,
        session_size: int = 6 # значение по умолчанию: 6
    ):
        """
        Инициализация MemoryManager.
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        self.session_size = session_size
        self.session_history: List[Dict[str, str]] = []

        # Инициализация ChromaDB
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(name="memory")
            logger.info(f"📂 Память подключена: {db_path}")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации ChromaDB: {e}")
            raise e

        # Ленивая загрузка модели (или использование существующей)
        self._load_model()

    def _load_model(self):
        """Загружает модель эмбеддингов, если она еще не загружена."""
        if MemoryManager._model_instance is None:
            logger.info(f"⏳ Загрузка модели эмбеддингов: {self.embedding_model_name}...")
            MemoryManager._model_instance = SentenceTransformer(self.embedding_model_name)
            logger.info("✅ Модель загружена.")
        self.model = MemoryManager._model_instance

    def _generate_id(self, text: str) -> str:
        """Генерирует стабильный ID на основе хеша текста."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    # ------------------ Долговременная память ------------------
    def delete_fact_by_query(self, query: str, n_results: int = 1) -> int:
        """
        Ищет релевантный факт(ы) по запросу и удаляет их.
        Возвращает количество удаленных документов.
        """
        if not query or self.collection.count() == 0:
            return 0
            
        try:
            query_emb = self.model.encode([query], show_progress_bar=False)[0].tolist()
            
            # 1. Сначала ищем релевантные ID
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=n_results,
                include=["documents"]
            )
            
            # Находим ID документов, которые нужно удалить
            # Поскольку ID генерируется через SHA256 от текста, 
            # мы можем использовать найденный текст для перегенерации ID, 
            # чтобы быть уверенными в удалении точного факта.
            ids_to_delete = []
            if results.get("documents", []) and results["documents"][0]:
                for doc_text in results["documents"][0]:
                    ids_to_delete.append(self._generate_id(doc_text))

            if not ids_to_delete:
                return 0
                
            # 2. Удаляем найденные ID
            self.collection.delete(ids=ids_to_delete)
            logger.warning(f"🗑️ Удалено {len(ids_to_delete)} фактов по запросу: {query}")
            return len(ids_to_delete)
            
        except Exception as e:
            logger.error(f"❌ Ошибка удаления памяти: {e}")
            return 0
    
    def remember(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Сохраняет или обновляет факт в долговременной памяти.
        Использует upsert для избежания дубликатов.
        """
        if not text or not text.strip():
            return False

        text = text.strip()
        doc_id = self._generate_id(text)
        
        try:
            # Генерируем эмбеддинг
            emb = self.model.encode([text], show_progress_bar=False)[0].tolist() # tolist() важен для сериализации
            
            # Используем upsert (вставка или обновление)
            self.collection.upsert(
                ids=[doc_id],
                documents=[text],
                embeddings=[emb],
                metadatas=[metadata or {"source": "user"}]
            )
            logger.info(f"💾 Запомнил: {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения в память: {e}")
            return False

    def recall(self, query: str, n_results: int = None) -> List[str]:
        """
        Возвращает релевантные факты.
        """
        if not query or not query.strip():
            return []
        
        # Если коллекция пуста, не тратим ресурсы на модель
        if self.collection.count() == 0:
            return []

        limit = n_results or self.top_k

        try:
            query_emb = self.model.encode([query], show_progress_bar=False)[0].tolist()
            
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=limit
            )
            
            # results["documents"] возвращает список списков [[doc1, doc2]]
            found_docs = results.get("documents", [])
            if found_docs and found_docs[0]:
                return found_docs[0]
            return []
            
        except Exception as e:
            logger.error(f"❌ Ошибка поиска в памяти: {e}")
            return []

    def wipe_memory(self):
        """Полная очистка базы знаний (Осторожно!)."""
        try:
            self.client.delete_collection("memory")
            self.collection = self.client.get_or_create_collection("memory")
            logger.warning("🧹 Память полностью очищена.")
        except Exception as e:
            logger.error(f"Ошибка очистки памяти: {e}")

    # ------------------ Короткая сессия ------------------

    def add_to_session(self, role: str, content: str):
        """Добавляет сообщение в историю сессии."""
        if not content:
            return
        self.session_history.append({"role": role, "content": content})
        if len(self.session_history) > self.session_size:
            self.session_history = self.session_history[-self.session_size:]

    def get_session_history(self) -> List[Dict[str, str]]:
        return self.session_history
    
    def clear_session(self):
        self.session_history = []

    # ------------------ Подготовка промпта ------------------

    def build_prompt(self, user_query: str, system_prompt: Optional[str] = None) -> str:
        """Формирует контекстный промпт."""
        # 1. Поиск фактов
        facts = self.recall(user_query)
        
        # 2. Сборка истории
        session_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in self.session_history])

        # 3. Сборка финального текста
        parts = []
        
        if system_prompt:
            parts.append(f"### SYSTEM INSTRUCTIONS\n{system_prompt}")
        
        if facts:
            facts_str = "\n".join([f"- {f}" for f in facts])
            parts.append(f"### LONG-TERM MEMORY (CONTEXT)\n{facts_str}")
            
        if session_text:
            parts.append(f"### DIALOGUE HISTORY\n{session_text}")
            
        parts.append(f"### USER QUERY\n{user_query}")

        return "\n\n".join(parts)

# ------------------ Тест ------------------
if __name__ == "__main__":
    # Настройка простого логирования для теста
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Инициализация менеджера памяти...")
    memory = MemoryManager()

    # Тест сохранения
    print("\n--- Тест сохранения ---")
    memory.remember("Пользователь использует macOS Ventura", {"category": "tech"})
    memory.remember("Предпочитаемый язык программирования: Python", {"category": "tech"})
    # Проверка дубликата (не должен упасть, должен обновить)
    memory.remember("Предпочитаемый язык программирования: Python", {"category": "tech_update"})

    # Тест поиска
    print("\n--- Тест поиска ---")
    query = "Какой комп у юзера?"
    facts = memory.recall(query)
    print(f"Запрос: {query}")
    print(f"Найдено: {facts}")

    # Тест промпта
    print("\n--- Тест генерации промпта ---")
    memory.add_to_session("user", "Привет")
    memory.add_to_session("ai", "Привет! Чем помочь?")
    
    final_prompt = memory.build_prompt("Напиши скрипт hello world", system_prompt="Ты кодер.")
    print("-" * 20)
    print(final_prompt)
    print("-" * 20)