# Autonomous AI Agent

**v0.5b**

Автономный CLI-агент на базе LangGraph с durable checkpointing, session resume, policy-driven tool execution, approval gate для опасных действий, MCP-интеграцией и встроенным `critic`-узлом для проверки завершённости задачи.

---

## Что нового в `0.5b`

- Добавлен настраиваемый checkpoint backend: `sqlite`, `memory`, `postgres`.
- Локальный CLI по умолчанию использует `sqlite`, поэтому сессии реально переживают перезапуск приложения.
- Появился `SessionStore`: активная сессия автоматически восстанавливается при следующем запуске.
- Для mutating/destructive tools добавлен approval flow на базе LangGraph `interrupt`.
- Реестр инструментов теперь хранит metadata/policy для каждого tool: `read_only`, `mutating`, `destructive`, `requires_approval`, `networked`.
- Добавлено локальное JSONL-логирование рантайма: tool calls, critic verdicts, retries, approvals, ошибки.
- `stop_background_process` по умолчанию не завершает внешние процессы, которые агент не запускал сам.
- `context7` включён в `mcp.json` как рекомендуемый MCP для документации и setup-задач.

---

## Возможности

### CLI и UX
- Потоковый вывод ответов и статусов инструментов.
- CLI на `rich` + `prompt_toolkit`.
- Fuzzy autocomplete для команд и путей.
- Автовосстановление последней активной сессии.
- Команда `/session` для просмотра текущего session/thread/checkpoint backend.

### Граф агента
- Основной поток: `summarize -> update_step -> agent -> approval/tools/critic`.
- `critic` не даёт агенту преждевременно объявить задачу завершённой.
- Автосжатие контекста по `SESSION_SIZE` с сохранением последних сообщений.
- Loop guard на уровне графа и на уровне повторяющихся tool-вызовов.
- Interrupt-driven approval для опасных действий без ручной перестройки графа.

### Инструменты и политика безопасности
- Filesystem tools для чтения, записи, редактирования, поиска и удаления файлов.
- Search tools через Tavily.
- System tools для диагностики ОС и сети.
- Process tools для фоновых процессов с ограничениями на `cwd` и внешний PID control.
- Опциональный shell tool с approval gate.
- MCP tools из `mcp.json`, включая `context7` и `sequential-thinking`.
- Policy metadata для каждого инструмента: read-only tools могут выполняться без approval, destructive tools требуют явного подтверждения.

### Надёжность и observability
- Durable checkpoints через SQLite/Postgres backend.
- Session snapshot в отдельном JSON-файле.
- JSONL run logs по сессиям.
- Явный runtime status по backend и MCP-серверам.
- Обработка malformed tool calls и protocol errors без краша CLI.

---

## Текущая структура проекта

```text
.
├── core/
│   ├── checkpointing.py
│   ├── cli_utils.py
│   ├── config.py
│   ├── constants.py
│   ├── errors.py
│   ├── fuzzy_completer.py
│   ├── logging_config.py
│   ├── message_utils.py
│   ├── nodes.py
│   ├── run_logger.py
│   ├── safety_policy.py
│   ├── session_store.py
│   ├── session_utils.py
│   ├── state.py
│   ├── stream_processor.py
│   ├── text_utils.py
│   ├── tool_policy.py
│   ├── tool_results.py
│   ├── ui_theme.py
│   ├── utils.py
│   └── validation.py
├── tools/
│   ├── delete_tools.py
│   ├── filesystem.py
│   ├── local_shell.py
│   ├── process_tools.py
│   ├── search_tools.py
│   ├── system_tools.py
│   └── tool_registry.py
├── tests/
├── agent.py
├── agent_cli.py
├── env_example.txt
├── mcp.json
├── prompt.txt
├── README.md
└── requirements.txt
```

---

## Назначение ключевых модулей

### `core/`
- `checkpointing.py` создаёт runtime checkpointer и выбирает backend (`sqlite`, `memory`, `postgres`).
- `config.py` загружает `.env`, парсит feature flags, лимиты, backend persistence и approval settings.
- `nodes.py` содержит LangGraph-узлы: `agent`, `approval`, `tools`, `critic`, summarize и runtime overlays.
- `state.py` описывает расширенное состояние графа, включая `session_id`, `run_id`, `pending_approval`, `last_tool_error`, `last_tool_result`.
- `session_store.py` хранит snapshot активной сессии для auto-resume.
- `run_logger.py` пишет JSONL events по сессиям.
- `tool_policy.py` описывает policy metadata для tool layer.
- `tool_results.py` нормализует tool output во внутренний structured shape для логирования и анализа.
- `stream_processor.py` обрабатывает поток LangGraph событий, tool results и interrupts.
- `session_utils.py`, `validation.py`, `errors.py`, `safety_policy.py` отвечают за repair, post-tool validation и security contracts.

### `tools/`
- `filesystem.py` реализует filesystem-инструменты с path guards, virtual mode и диффами для edit.
- `search_tools.py` даёт web search/fetch через Tavily.
- `system_tools.py` отдаёт системную и сетевую информацию.
- `process_tools.py` управляет фоновыми процессами и не позволяет по умолчанию убивать произвольные внешние PID.
- `local_shell.py` содержит shell tool с общими лимитами и approval-политикой.
- `tool_registry.py` собирает локальные и MCP-инструменты, назначает им metadata и формирует runtime status.

---

## Конфигурация `.env`

### Провайдер модели

| Параметр | Назначение |
| :--- | :--- |
| `PROVIDER` | `gemini` или `openai` |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` | ключ выбранного провайдера |
| `GEMINI_MODEL` / `OPENAI_MODEL` | имя модели |
| `OPENAI_BASE_URL` | base URL для OpenAI-совместимых API |

### Runtime и persistence

| Параметр | Значение по умолчанию | Назначение |
| :--- | :--- | :--- |
| `CHECKPOINT_BACKEND` | `sqlite` | backend для checkpoint saver |
| `CHECKPOINT_SQLITE_PATH` | `.agent_state/checkpoints.sqlite` | локальная SQLite БД для state persistence |
| `CHECKPOINT_POSTGRES_URL` | пусто | строка подключения к Postgres backend |
| `SESSION_STATE_PATH` | `.agent_state/session.json` | snapshot активной сессии |
| `RUN_LOG_DIR` | `logs/runs` | директория JSONL run logs |

### Поведение агента

| Параметр | Значение по умолчанию |
| :--- | :--- |
| `TEMPERATURE` | `0.2` |
| `MAX_LOOPS` | `50` |
| `PROMPT_PATH` | `prompt.txt` |
| `MODEL_SUPPORTS_TOOLS` | `true` |
| `DEBUG` | `false` |
| `STRICT_MODE` | `false` |

### Включение подсистем

| Параметр | Значение по умолчанию |
| :--- | :--- |
| `ENABLE_FILESYSTEM_TOOLS` | `true` |
| `ENABLE_SEARCH_TOOLS` | `true` |
| `ENABLE_SYSTEM_TOOLS` | `true` |
| `ENABLE_PROCESS_TOOLS` | `false` |
| `ENABLE_SHELL_TOOL` | `false` |
| `ENABLE_APPROVALS` | `true` |
| `ALLOW_EXTERNAL_PROCESS_CONTROL` | `false` |

### Лимиты и безопасность

| Параметр | Значение по умолчанию | Комментарий |
| :--- | :--- | :--- |
| `MAX_TOOL_OUTPUT` | `10000` | лимит вывода инструмента |
| `MAX_SEARCH_CHARS` | `15000` | лимит search/fetch контента |
| `MAX_FILE_SIZE` | `300MiB` | число без суффикса = байты |
| `MAX_READ_LINES` | `2000` | лимит строк при чтении |
| `MAX_BACKGROUND_PROCESSES` | `5` | лимит фоновых задач |

Для `MAX_FILE_SIZE` используйте явные единицы: `4MB`, `300MiB`, `1GiB`. Значение `300` означает `300 bytes`, а не `300 MB`.

---

## MCP

- Конфигурация серверов находится в `mcp.json`.
- Подключаются только записи с `"enabled": true`.
- MCP tools регистрируются вместе с локальными инструментами и получают runtime status-report.
- По умолчанию в конфиге включены:
  - `context7` для документации, setup и API reference.
  - `sequential-thinking` для сложных reasoning-задач.

---

## Установка и запуск

### 1. Подготовка окружения

Требуется Python `3.10+`. Для некоторых MCP-серверов может понадобиться Node.js.

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

`requirements.txt` включает:
- `langgraph-checkpoint-sqlite` для локального durable runtime;
- `langgraph-checkpoint-postgres` и `psycopg` для production-like backend.

### 3. Настройка `.env`

```powershell
Copy-Item env_example.txt .env
```

После этого:
- укажите ключи модели;
- при необходимости добавьте `TAVILY_API_KEY`;
- выберите checkpoint backend;
- при необходимости включите process/shell tools;
- при необходимости настройте MCP-серверы в `mcp.json`.

### 4. Запуск

```bash
python agent_cli.py
```

При старте CLI:
- поднимет последнюю активную сессию из `SESSION_STATE_PATH`;
- покажет текущий checkpoint backend;
- выведет runtime status по MCP и backend’ам.

---

## Тесты

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Покрываются:
- critic/graph flow;
- filesystem и stream processor;
- checkpoint runtime и SQLite persistence;
- approval interrupts и resume flow;
- session store;
- JSONL run logging;
- safer process control.

---

## Команды в CLI

- `/help` — краткая справка
- `/tools` — список доступных инструментов
- `/session` — текущая сессия, thread и checkpoint backend
- `clear` / `reset` — создать новую сессию
- `exit` / `quit` — выход
- `Alt+Enter` — многострочный ввод

---

## Примечания по безопасности

- Read-only tools могут выполняться автономно.
- Mutating/destructive tools по умолчанию требуют approval.
- `cli_exec` отключён по умолчанию и считается high-risk инструментом.
- `stop_background_process` не завершает внешние процессы без `ALLOW_EXTERNAL_PROCESS_CONTROL=true`.
- Для production-режима предпочтителен `postgres` backend; для локального CLI — `sqlite`.
