# Autonomous AI Agent

**v0.62.3b**

Автономный CLI-агент на базе LangGraph с durable checkpointing, session resume, policy-driven tool execution, approval gate для опасных действий, MCP-интеграцией, встроенным `critic`-узлом и локальным workflow-first TUI на `rich` + `prompt_toolkit`.

---

## Что нового в `v0.62.3b`

- Добавлен настраиваемый checkpoint backend: `sqlite`, `memory`, `postgres`.
- Локальный CLI по умолчанию использует `sqlite`, поэтому сессии переживают перезапуск приложения.
- Появился `SessionStore`: активная сессия автоматически восстанавливается при следующем запуске.
- Для mutating/destructive tools добавлен approval flow на базе LangGraph `interrupt`.
- Реестр инструментов хранит metadata/policy для каждого tool: `read_only`, `mutating`, `destructive`, `requires_approval`, `networked`.
- Добавлено локальное JSONL-логирование рантайма: tool calls, critic verdicts, retries, approvals, ошибки.
- Read-only инструменты запускаются параллельно через `asyncio.gather`.
- Добавлен loop guard с отдельными лимитами для read-only и mutating инструментов.
- TUI обновлён до более строгого monochrome-стиля: белый/серый каркас, один холодный синий акцент и shimmer-статусы для активных фаз.
- Ввод больше не подсвечивается как Markdown, а code blocks и diff preview сохраняют цветную syntax highlighting.

## Возможности

### CLI и UX
- Единая overview-панель при старте: провайдер, модель, backend, session/thread, approvals, число инструментов и активные MCP-серверы.
- Потоковый вывод ответов: спиннер и накапливающийся текст отображаются одновременно.
- Контекстный статус по активному узлу графа: `Thinking`, `Running tools`, `Reviewing`, `Waiting for approval`, `Compressing context`.
- Для рабочих фаз используется shimmer-анимация текста; approval-состояние остаётся статичным для читаемости.
- Тайминг каждого tool-вызова: `tool ▶ name(...)` при старте, `tool ✔ summary 1.4s` при завершении.
- Approval-промпт с risk summary и selector `Yes / No / Always`.
- `Always` сохраняется в active session snapshot и сбрасывается через `/new`.
- Контекстный bottom toolbar и команды `/help`, `/tools`, `/session`, `/new`, `/quit`.
- Fuzzy autocomplete для команд и путей.
- Каркас интерфейса придерживается схемы white/gray + one-accent-blue; syntax highlighting для кода и diff рендерится отдельно.

### Граф агента
- Базовый поток: `summarize -> update_step -> agent -> approval/tools/critic`.
- `critic` не даёт агенту преждевременно объявить задачу завершённой.
- `critic` работает в двух режимах:
  - `critic_source=agent` проверяет полноту текстового ответа.
  - `critic_source=tools` проверяет, закрывают ли результаты инструментов исходную задачу.
- Автосжатие контекста запускается по `SESSION_SIZE`, сохраняя последние `SUMMARY_KEEP_LAST` сообщений без сжатия.
- Interrupt-driven approval встроен в граф без ручной перестройки основного workflow.

### Инструменты и безопасность
- Filesystem tools для чтения, записи, редактирования, поиска и удаления файлов.
- Search tools через Tavily.
- System tools для диагностики ОС и сети.
- Process tools для фоновых процессов с ограничениями на внешний PID control.
- Опциональный shell tool с approval gate.
- MCP tools из `mcp.json`.
- Политика на уровне инструмента: read-only, mutating, destructive, approval-required и networked.

### Надёжность и observability
- Durable checkpoints через SQLite/Postgres backend.
- Session snapshot в отдельном JSON-файле.
- JSONL run logs по сессиям.
- Runtime status-report по backend, local loaders и MCP-серверам.
- Обработка malformed tool calls и protocol errors без краша CLI.

---

## Актуальная архитектура

### Точка сборки
- [agent.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/agent.py) создаёт LLM, `ToolRegistry`, checkpoint runtime и `JsonlRunLogger`, затем компилирует LangGraph workflow.
- [agent_cli.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/agent_cli.py) отвечает за локальный CLI, overview, slash-команды, approval UX и запуск потокового рендера.

### `core/`
- [core/config.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/core/config.py) загружает `.env`, парсит runtime path fields, feature flags, summarization thresholds, loop-guard настройки, retry settings и approval settings.
- [core/checkpointing.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/core/checkpointing.py) создаёт runtime checkpointer и выбирает backend `sqlite` / `memory` / `postgres`.
- [core/nodes.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/core/nodes.py) содержит LangGraph-узлы: `summarize`, `agent`, `approval`, `tools`, `critic`.
- [core/state.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/core/state.py) описывает расширенное состояние графа, включая `session_id`, `run_id`, `critic_*`, `pending_approval`, `last_tool_error`, `last_tool_result`.
- [core/session_store.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/core/session_store.py) хранит snapshot активной сессии для auto-resume, включая session-scoped approval mode.
- [core/run_logger.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/core/run_logger.py) пишет JSONL events по сессиям.
- [core/tool_policy.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/core/tool_policy.py) описывает metadata/policy для tool layer.
- [core/tool_results.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/core/tool_results.py) нормализует tool output для логирования и анализа.
- [core/stream_processor.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/core/stream_processor.py) обрабатывает поток LangGraph событий, tool results и approval interrupts.
- [core/ui_theme.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/core/ui_theme.py) хранит централизованную палитру TUI и shimmer-хелперы.
- [core/session_utils.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/core/session_utils.py), [core/validation.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/core/validation.py), [core/errors.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/core/errors.py), [core/safety_policy.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/core/safety_policy.py) отвечают за repair, post-tool validation и security contracts.

### `tools/`
- [tools/tool_registry.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/tools/tool_registry.py) собирает локальные и MCP-инструменты, назначает им metadata и формирует runtime status.
- [tools/filesystem.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/tools/filesystem.py) реализует filesystem-инструменты.
- [tools/filesystem_impl/manager.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/tools/filesystem_impl/manager.py), [tools/filesystem_impl/editing.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/tools/filesystem_impl/editing.py), [tools/filesystem_impl/pathing.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/tools/filesystem_impl/pathing.py) содержат низкоуровневую реализацию файловых операций.
- [tools/search_tools.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/tools/search_tools.py) даёт web search/fetch через Tavily.
- [tools/system_tools.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/tools/system_tools.py) отдаёт системную и сетевую информацию.
- [tools/process_tools.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/tools/process_tools.py) управляет фоновыми процессами и ограничивает контроль внешних PID.
- [tools/local_shell.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/tools/local_shell.py) содержит shell tool с approval-политикой.
- [tools/delete_tools.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/tools/delete_tools.py) используется как fallback для удаления, если filesystem-tools отключены.

### Дополнительно
- [pdf_mcp/pdf_server.py](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/pdf_mcp/pdf_server.py) содержит отдельный MCP-сервер для PDF-задач.
- [tests](/D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/tests) покрывают граф, CLI UX, filesystem, checkpointing, tooling refactor и PDF MCP.

---

## Текущая структура проекта

```text
.
├── core/
│   ├── __init__.py
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
├── pdf_mcp/
│   ├── fonts/
│   ├── pdf_server.py
│   └── README.md
├── tests/
│   ├── test_cli_ux.py
│   ├── test_critic_graph.py
│   ├── test_pdf_mcp.py
│   ├── test_runtime_refactor.py
│   ├── test_stream_and_filesystem.py
│   └── test_tooling_refactor.py
├── tools/
│   ├── __init__.py
│   ├── delete_tools.py
│   ├── filesystem.py
│   ├── filesystem_impl/
│   │   ├── __init__.py
│   │   ├── editing.py
│   │   ├── manager.py
│   │   └── pathing.py
│   ├── local_shell.py
│   ├── process_tools.py
│   ├── search_tools.py
│   ├── system_tools.py
│   └── tool_registry.py
├── agent.py
├── agent_cli.py
├── build.bat
├── check_models.py
├── env_example.txt
├── icon.ico
├── mcp.json
├── models_comparison.md
├── prompt.txt
├── README.md
└── requirements.txt
```

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
| `PROMPT_PATH` | `prompt.txt` | путь к системному промпту |
| `MCP_CONFIG_PATH` | `mcp.json` | путь к конфигурации MCP-серверов |
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
| `MODEL_SUPPORTS_TOOLS` | `true` |
| `DEBUG` | `false` |
| `STRICT_MODE` | `false` |
| `SESSION_SIZE` | `8000` |
| `SUMMARY_KEEP_LAST` | `4` |
| `MAX_RETRIES` | `3` |
| `RETRY_DELAY` | `2` |

### Loop guard для инструментов

| Параметр | Значение по умолчанию | Назначение |
| :--- | :--- | :--- |
| `TOOL_LOOP_WINDOW` | вычисляется от `MAX_LOOPS` | окно истории для поиска повторяющихся tool calls |
| `TOOL_LOOP_LIMIT_MUTATING` | вычисляется от `MAX_LOOPS` | лимит повторов для mutating tools |
| `TOOL_LOOP_LIMIT_READONLY` | вычисляется от `MAX_LOOPS` | лимит повторов для read-only tools |

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
| `MAX_TOOL_OUTPUT` | `4000` | лимит вывода инструмента |
| `MAX_SEARCH_CHARS` | `15000` | лимит search/fetch контента |
| `MAX_FILE_SIZE` | `300MiB` | число без суффикса трактуется как байты |
| `MAX_READ_LINES` | `2000` | лимит строк при чтении |
| `MAX_BACKGROUND_PROCESSES` | `5` | лимит фоновых задач |

Для `MAX_FILE_SIZE` лучше указывать явные единицы: `4MB`, `300MiB`, `1GiB`.

---

## MCP

- Конфигурация серверов находится в `mcp.json`.
- Подключаются только записи с `"enabled": true`.
- MCP tools регистрируются вместе с локальными инструментами и получают runtime status-report.
- Набор активных серверов определяется полем `"enabled"` в `mcp.json`, а не кодом агента.
- В поставляемом `mcp.json` обычно используются `context7` и `sequential-thinking`.

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
- покажет overview по модели, backend, session/thread, approvals и MCP;
- выведет краткий cheatsheet по slash-командам.

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
- safer process control;
- CLI UX: overview, tools/help panels, stream layout, shimmer/status rendering и prompt palette.

Примечание:
- полный прогон может упереться во внешнюю проблему окружения вокруг `pdf_mcp`/`fitz`; это не относится к архитектуре CLI/TUI.

---

## Команды в CLI

- `/help` — краткая справка
- `/tools` — список инструментов с группировкой и policy badges
- `/session` — текущий runtime overview
- `/new` — создать новую сессию
- `clear` / `reset` — альтернативный reset
- `/quit` — выход
- `Alt+Enter` — многострочный ввод

---

## Примечания по безопасности

- Read-only tools могут выполняться автономно.
- Mutating/destructive tools по умолчанию требуют approval.
- Approval в CLI выбирается через selector `Yes / No / Always`, без ручного `y/n` ввода.
- Режим `Always` распространяется на все последующие protected actions в рамках текущей session snapshot и сбрасывается через `/new`.
- `Esc` и `Ctrl+C` в approval selector трактуются как безопасный отказ.
- При отказе пользователя (`ACCESS_DENIED`) агент не симулирует и не выдумывает результат.
- `cli_exec` отключён по умолчанию и считается high-risk инструментом.
- `stop_background_process` не завершает внешние процессы без `ALLOW_EXTERNAL_PROCESS_CONTROL=true`.
- Для production-режима предпочтителен `postgres` backend; для локального CLI — `sqlite`.
