# 🤖 AI Agent (LangGraph + Долговременная Память)

Этот проект — универсальный **AI-агент** на базе **LangGraph** и **Model Context Protocol (MCP)** с полноценной **долговременной памятью**. Работает в **CLI** (терминал) и **Web UI** (Streamlit).

---

## ✨ Возможности

- **Архитектура:** LangGraph для циклов (планирование → инструменты → анализ → исправление ошибок).
- **LLM:** Google Gemini / OpenAI-совместимые API (Grok, etc.).
- **Память (LTM):** ChromaDB + мультиязычные эмбеддинги. Авто `remember_fact` / `recall_facts` / `delete_facts`.
- **Токены:** `SESSION_SIZE` для лимита истории (экономия токенов).
- **Инструменты:** MCP (файлы, поиск, CLI) + локальные (`safe_delete`).
- **Интерфейсы:** CLI (`rich` + `prompt_toolkit`) / Web (`Streamlit`).
- **Контекст:** MemorySaver для диалогов.

---

## 🚀 Установка

1. Клонируйте/скачайте проект.

2. Создайте venv:
   
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. Установите зависимости:
   
   ```bash
   pip install -r requirements.txt
   ```

4. Настройте `.env` (копия из `env_example`):
   
   ```ini
   # Основные
   PROVIDER=gemini  # или openai
   GEMINI_API_KEY=your_key
   GEMINI_MODEL=gemini-2.0-flash-exp
   
   # OpenAI-совместимые
   # PROVIDER=openai
   # OPENAI_API_KEY=your_key
   # OPENAI_MODEL=x-ai/grok-4.1-fast:free
   # OPENAI_BASE_URL=https://openrouter.ai/api/v1
   
   # Память & Токены
   LONG_TERM_MEMORY=true
   SESSION_SIZE=6  # Лимит сообщений в сессии
   
   # Отладка
   DEBUG=false
   MAX_RETRIES=3
   RETRY_DELAY=2
   ```

---

## 🏃‍♂️ Запуск

Единый скрипт `start.py`:

```bash
python start.py  # Меню (CLI/UI)
python start.py cli  # Терминал
python start.py ui   # Web (http://localhost:8501)
```

### MCP Конфиг (`mcp.json`)

```json
{
  "filesystem": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "{filesystem_path}"],
    "transport": "stdio",
    "enabled": true
  }
}
```

---

## 📁 Структура проекта

| Файл/Директория     | Описание                                     |
| ------------------- | -------------------------------------------- |
| `agent.py`          | Ядро агента (LangGraph + инструменты).       |
| `memory_manager.py` | Долговременная память (ChromaDB).            |
| `prompt.md`         | Системный промпт (правила МЫСЛЬ → ДЕЙСТВИЕ). |
| `agent_cli.py`      | CLI-интерфейс (`rich`/`prompt_toolkit`).     |
| `ui.py`             | Web UI (Streamlit).                          |
| `start.py`          | Запуск (CLI/UI).                             |
| `delete_tools.py`   | Локальные инструменты (`safe_delete`).       |
| `logging_config.py` | Логи (фильтр шума).                          |
| `.env`              | Конфиг (API-ключи, память).                  |
| `mcp.json`          | MCP-сервера.                                 |
| `requirements.txt`  | Зависимости.                                 |
| `memory_db/`        | База памяти (ChromaDB).                      |

---

## 🔧 Дополнительно

- **Логи:** `ai_agent.log`.
- **Память:** Авто-работа с фактами (категории: `AI models`, `md format rules`).
- **Примеры:** См. `new1.md` (AI-модели).

⭐ **Star на GitHub** / Пожертвования: [Купить кофе](https://ko-fi.com/your-link).

--- 

*© 2025. Open-source под MIT.*