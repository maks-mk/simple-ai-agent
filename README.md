# Autonomous AI Agent (LangGraph + MCP + Rich UI)

**v7.43b**

Автономный AI-агент на базе **LangGraph** и **MCP (Model Context Protocol)** с CLI-интерфейсом для Windows/Linux.

---

## Основные возможности

### CLI и UI
- **Smart Formatting:** компактное отображение вызовов инструментов, сокращение длинных аргументов, подсветка ключевых данных.
- **Real-time Streaming:** потоковый вывод ответов и tool-статусов.
- **Rich UI:** консольный интерфейс на `rich`.
- **Fuzzy Autocomplete:** нечеткое автодополнение путей и команд.
- **Session Recovery:** восстановление сессии после сбоев.

### Архитектура и логика
- **LangGraph Core:** `Start -> Summarize -> UpdateStep -> Agent -> (Tools -> UpdateStep -> Agent | End)`.
- **Memory Management:** автосуммаризация при достижении `SESSION_SIZE` с сохранением последних `SUMMARY_KEEP_LAST` сообщений.
- **Loop Guard (graph):** ограничение шагов по `MAX_LOOPS`.
- **Loop Guard (tools):** защита от повторов одинаковых вызовов инструментов; дефолты синхронизированы с `MAX_LOOPS`, но могут быть переопределены в `.env`.
- **Parallel Tool Calls:** параллельный запуск только для безопасного набора read-only инструментов.
- **Token Tracking:** учет usage metadata с fallback-логикой.

### Инструменты (модульно через `.env`)
1. **Filesystem Tools** (`ENABLE_FILESYSTEM_TOOLS`):
   - `read_file`, `write_file`, `edit_file`, `list_directory`
   - `search_in_file`, `search_in_directory`, `find_file`, `tail_file`
   - `download_file`, `safe_delete_file`, `safe_delete_directory`
   - защита от path traversal
2. **Search Tools** (`ENABLE_SEARCH_TOOLS`):
   - `web_search(query, max_results=5, search_depth="basic", topic="general")`
   - `fetch_content(urls, advanced=False, content_format="markdown", query=None, chunks_per_source=3)`
   - `batch_web_search(queries, max_results=5, search_depth="basic", topic="general")`
   - `crawl_site` (опционально, если доступен в окружении)
3. **System Tools** (`ENABLE_SYSTEM_TOOLS`):
   - `get_system_info`, `get_public_ip`, `lookup_ip_info`, `get_local_network_info`
4. **Shell Tool** (`ENABLE_SHELL_TOOL`):
   - `cli_exec` (timeout + safety policy, с подсказками Windows для Unix-команд)
5. **Process Tools** (`ENABLE_PROCESS_TOOLS`):
   - `run_background_process`, `stop_background_process`, `find_process_by_port`

### MCP интеграция
- Конфигурация: `mcp.json`
- Загружаются только серверы с `"enabled": true`
- MCP-инструменты подключаются асинхронно и параллельно (с ограничением конкурентности)

---

## Конфигурация (.env)

### Провайдеры LLM

| Параметр | Описание | Пример |
| :--- | :--- | :--- |
| `PROVIDER` | `gemini` или `openai` | `openai` |
| `GEMINI_API_KEY` | API ключ Gemini | `AIzaSy...` |
| `GEMINI_MODEL` | Модель Gemini | `gemini-1.5-flash` |
| `OPENAI_API_KEY` | API ключ OpenAI/совместимых API | `sk-...` |
| `OPENAI_BASE_URL` | Base URL совместимого API | `https://openrouter.ai/api/v1` |
| `OPENAI_MODEL` | Имя модели OpenAI/совместимой | `gpt-4o` |

### Основные настройки

| Параметр | По умолчанию | Описание |
| :--- | :--- | :--- |
| `TEMPERATURE` | `0.2` | Температура модели |
| `MAX_LOOPS` | `50` | Максимум шагов графа на запрос |
| `TOOL_LOOP_LIMIT_MUTATING` | `auto` | Лимит повторов mutating tools (дефолт: `max(3, min(8, MAX_LOOPS // 10))`) |
| `TOOL_LOOP_LIMIT_READONLY` | `auto` | Лимит повторов read-only tools (дефолт: `max(6, min(20, MAX_LOOPS // 4))`) |
| `TOOL_LOOP_WINDOW` | `auto` | Окно истории для проверки повторов (дефолт: `max(10, min(60, MAX_LOOPS))`) |
| `DEBUG` | `false` | Расширенные логи |
| `PROMPT_PATH` | `prompt.txt` | Путь к системному промпту |
| `MODEL_SUPPORTS_TOOLS` | `true` | Разрешить tool-calling (иначе chat-only) |

### Управление инструментами

| Параметр | По умолчанию | Описание |
| :--- | :--- | :--- |
| `ENABLE_SEARCH_TOOLS` | `true` | Включить search tools |
| `TAVILY_API_KEY` | - | Ключ Tavily |
| `ENABLE_FILESYSTEM_TOOLS` | `true` | Включить filesystem tools |
| `ENABLE_SHELL_TOOL` | `false` | Включить `cli_exec` |
| `ENABLE_SYSTEM_TOOLS` | `true` | Включить system tools |
| `ENABLE_PROCESS_TOOLS` | `false` | Включить process tools |

### Search API детали (Tavily)

- `web_search`:
  - `max_results`: `1..20`
  - `search_depth`: `basic | advanced | fast | ultra-fast`
  - `topic`: `general | news | finance`
- `fetch_content`:
  - поддерживает до `20` URL за вызов
  - `content_format`: `markdown | text`
  - при `query` можно задавать `chunks_per_source` (`1..20`)
- `batch_web_search`:
  - выполняет до `5` уникальных запросов параллельно
  - использует те же параметры `max_results/search_depth/topic`, что и `web_search`
- Ошибки Tavily приводятся к типизированным ошибкам агента (`CONFIG`, `ACCESS_DENIED`, `LIMIT_EXCEEDED`, `TIMEOUT`, `NETWORK`).

### Безопасность и лимиты

| Параметр | По умолчанию | Описание |
| :--- | :--- | :--- |
| `MAX_TOOL_OUTPUT` | `4000` | Лимит вывода инструментов |
| `MAX_SEARCH_CHARS` | `15000` | Лимит контента search/fetch |
| `MAX_FILE_SIZE` | `300MB` | Максимальный размер файла |
| `MAX_READ_LINES` | `2000` | Максимум строк при чтении |
| `MAX_BACKGROUND_PROCESSES` | `5` | Лимит фоновых процессов |

### Память и retry

| Параметр | По умолчанию | Описание |
| :--- | :--- | :--- |
| `SESSION_SIZE` | `8000` | Порог суммаризации контекста |
| `SUMMARY_KEEP_LAST` | `4` | Сколько последних сообщений не сжимать |
| `MAX_RETRIES` | `3` | Повторы сетевых запросов |
| `RETRY_DELAY` | `2` | Задержка между повторами |

---

## Структура проекта

```text
.
├── core/
│   ├── config.py
│   ├── nodes.py
│   ├── state.py
│   ├── stream_processor.py
│   ├── cli_utils.py
│   ├── ui_theme.py
│   ├── fuzzy_completer.py
│   ├── logging_config.py
│   ├── constants.py
│   └── session_utils.py
├── tools/
│   ├── filesystem.py
│   ├── delete_tools.py
│   ├── search_tools.py
│   ├── system_tools.py
│   ├── process_tools.py
│   ├── local_shell.py
│   └── tool_registry.py
├── agent.py
├── agent_cli.py
├── mcp.json
├── requirements.txt
├── .env
├── env_example.txt
└── prompt.txt
```

---

## Архитектурные детали

1. `AgentConfig` загружает и валидирует `.env`.
2. `ToolRegistry` подключает локальные и MCP-инструменты.
3. `build_agent_app()` собирает граф с узлами `summarize`, `update_step`, `agent`, `tools`.
4. Завершение происходит при:
   - достижении `MAX_LOOPS`;
   - отсутствии tool-вызовов от модели;
   - chat-only режиме (`MODEL_SUPPORTS_TOOLS=false`).

---

## Установка и запуск

### 1. Подготовка окружения
Нужен **Python 3.10+**. Для MCP-серверов типа `npx` может потребоваться **Node.js**.

```bash
python -m venv venv
```

Активация:
- Windows: `venv\Scripts\activate`
- Linux/macOS: `source venv/bin/activate`

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Настройка

1. Скопируйте шаблон:
   - Windows (PowerShell): `Copy-Item env_example.txt .env`
   - Linux/macOS: `cp env_example.txt .env`
2. Заполните ключи в `.env` (`OPENAI_API_KEY`/`GEMINI_API_KEY`, при необходимости `TAVILY_API_KEY`).
3. При необходимости включите MCP-серверы в `mcp.json` через `"enabled": true`.

### 4. Запуск

```bash
python agent_cli.py
```

---

## Ограничения и безопасность

- `MAX_LOOPS` ограничивает только шаги графа.
- Per-tool loop guard работает отдельно и может остановить повторяющиеся вызовы раньше; по умолчанию его пороги вычисляются от `MAX_LOOPS`, либо задаются через `TOOL_LOOP_LIMIT_MUTATING`, `TOOL_LOOP_LIMIT_READONLY`, `TOOL_LOOP_WINDOW`.
- `cli_exec` потенциально опасен, по умолчанию выключен.
- `edit_file` содержит защитные проверки и дополнительную JSON-валидацию для `.json`.

---

## Команды в CLI

- `/help` — справка
- `/tools` — список доступных инструментов
- `clear` / `reset` — очистка сессии
- `exit` / `quit` — выход
- `Alt+Enter` — перенос строки в вводе


