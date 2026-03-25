# Autonomous AI Agent

**v0.62b**

Автономный CLI-агент на базе LangGraph с durable checkpointing, session resume, policy-driven tool execution, approval gate для опасных действий, MCP-интеграцией и встроенным `critic`-узлом для проверки завершённости задачи.

---

## Что нового в `v0.62b`

- Добавлен настраиваемый checkpoint backend: `sqlite`, `memory`, `postgres`.
- Локальный CLI по умолчанию использует `sqlite`, поэтому сессии реально переживают перезапуск приложения.
- Появился `SessionStore`: активная сессия автоматически восстанавливается при следующем запуске.
- Для mutating/destructive tools добавлен approval flow на базе LangGraph `interrupt`.
- Реестр инструментов теперь хранит metadata/policy для каждого tool: `read_only`, `mutating`, `destructive`, `requires_approval`, `networked`.
- Добавлено локальное JSONL-логирование рантайма: tool calls, critic verdicts, retries, approvals, ошибки.
- `stop_background_process` по умолчанию не завершает внешние процессы, которые агент не запускал сам.
- `context7` включён в `mcp.json` как рекомендуемый MCP для документации и setup-задач.
- Read-only инструменты запускаются параллельно через `asyncio.gather` — ускоряет батчевые операции чтения/поиска.
- Детектор зацикливания с раздельными лимитами для read-only и mutating инструментов.

## UI/UX улучшения (workflow-first refresh)

**`core/constants.py`**
- `AGENT_VERSION` вынесен как единый источник истины; хардкод версии из `agent_cli.py` удалён.
- Версия обновлена до `v0.61b`.

**`core/ui_theme.py`**
- Палитра стала более семантической: отдельно разведены нейтральный текст, tool/action-акценты, warning/danger и overview-метки.
- Добавлены стили `tool.readonly`, `tool.mcp`, `approval.summary`, `turn.assistant`, `overview.label/value`.

**`core/stream_processor.py`**
- Исправлен главный UX-баг: спиннер и частичный текст теперь отображаются одновременно через `RichGroup` — пользователь всегда видит, что агент работает.
- Спиннер показывает нормализованные статусы по активному узлу: `Thinking`, `Running tools`, `Reviewing`, `Waiting for approval`, `Compressing context`, с elapsed time.
- Отслеживание активного узла графа через поле `active_node`.
- Tool activity, summarization notes и diff preview теперь оформлены единообразно и визуально привязаны к текущему ходу агента.
- Тайминг каждого tool-вызова: `tool ▶ name(...)` при старте, `tool ✔ summary 1.4s` при завершении.

**`core/text_utils.py`**
- `_rewrite_local_file_links()`: локальные Markdown-ссылки `[имя.md](имя.md)` автоматически конвертируются в инлайн-код `` `имя.md` `` до передачи в Rich — устраняет URL-кодирование кириллицы в именах файлов (`%D0%BC%D0%BE%D0%B4...`).

**`core/nodes.py`**
- При `ACCESS_DENIED` агент получает жёсткий системный оверлей вместо общего `UNRESOLVED_TOOL_ERROR_PROMPT_TEMPLATE` — запрещает симулировать, эмулировать или выдумывать результат отклонённого вызова.
- Расширен список `failure_markers` в `_assistant_acknowledges_unresolved_tool_error`: добавлены `denied`, `access_denied`, `cancelled by`, `was cancelled`, `отказан`, `отклонён` — critic теперь правильно завершает цикл после явного отказа пользователя.

**`agent_cli.py`**
- Стартовые `header/runtime/session` объединены в единый overview-панель: провайдер, модель, backend, session/thread, approvals, tool count и активные MCP-серверы видны сразу.
- Под overview выводится компактный cheatsheet: `/help`, `/tools`, `/session`, `/new`, `Alt+Enter`.
- `get_bottom_toolbar`: контекстный — в обычном режиме показывает основные команды, а в approval prompt переключается на подсказки для подтверждения.
- `render_tools`: инструменты теперь группируются в `Read-only`, `Protected`, `MCP` и показывают policy badges (`read-only`, `approval`, `mutating`, `destructive`, `network`, `mcp`).
- `render_help`: теперь task-oriented — старт работы, инспекция инструментов и сессии, reset, approvals, multiline input.
- `initialize_agent`: прогресс-статус с именем провайдера и модели, `✔ Agent ready` после загрузки.
- `prompt_for_interrupt`: risk-first approval panel с summary по destructive/mutating/networked calls и компактным selector на `Yes` / `No` / `Always`.
- `Always` действует на все последующие protected actions в рамках текущей session snapshot, переживает auto-resume и сбрасывается через `/new`.
- При активном `Always` approval prompt больше не открывается — CLI печатает короткое `auto-approved` уведомление и продолжает выполнение.
- Главный цикл: тонкий `Rule()` между ходами, явные метки `You` / `Agent`, `✕ cancelled` при `Ctrl+C`, ошибки оборачиваются в `Panel(border_style="panel.error")`.
- Ошибки инициализации (`Config error`, `Init error`) тоже в styled Panel.
- Команда выхода приведена к slash-стилю: используется `/quit`.
- `_format_policy_flags()` вынесена в отдельную функцию.
- `cli_utils.py`: `Alt+Enter` по-прежнему добавляет новую строку в основном вводе; approval selector обрабатывается отдельно через `prompt_toolkit`.

---

## Возможности

### CLI и UX
- Единый overview-панель при старте: провайдер, модель, backend, session/thread, approvals, tool count и активные MCP-серверы.
- Потоковый вывод ответов: спиннер и накапливающийся текст отображаются одновременно — пользователь всегда видит прогресс.
- Контекстный спиннер по активному узлу графа: `Thinking` / `Running tools` / `Reviewing` / `Waiting for approval` / `Compressing context` с elapsed time.
- Тайминг каждого tool-вызова: `tool ▶ имя(...)` при старте, `tool ✔ summary 1.4s` при завершении.
- Approval-промпт с risk summary, цветными policy-флагами и компактным selector `Yes / No / Always`.
- `Always` сохраняется в активной session snapshot, показывается в overview как `on (always for this session)` и сбрасывается командой `/new`.
- Контекстный bottom toolbar: основные slash-команды в обычном режиме и отдельная подсказка в approval prompt.
- Явные turn labels `You` / `Agent` делают историю диалога легче для чтения.
- CLI на `rich` + `prompt_toolkit`; fuzzy autocomplete для команд и путей.
- Автовосстановление последней активной сессии.
- Команды `/help`, `/tools`, `/session`, `/new`, `/quit` для быстрого доступа к справке, состоянию и управлению сессией.

### Граф агента
- Основной поток: `summarize -> update_step -> agent -> approval/tools/critic`.
- `critic` не даёт агенту преждевременно объявить задачу завершённой. Работает в двух режимах: после ответа агента (`critic_source=agent`) — проверяет полноту текстового ответа; после выполнения инструментов (`critic_source=tools`) — проверяет что результаты инструментов действительно закрывают задачу.
- Автосжатие контекста по `SESSION_SIZE` с сохранением последних сообщений.
- Loop guard на уровне графа и на уровне повторяющихся tool-вызовов, с раздельными лимитами для read-only и mutating инструментов.
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
- `session_store.py` хранит snapshot активной сессии для auto-resume, включая session-scoped approval mode.
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
| `MAX_TOOL_OUTPUT` | `4000` | лимит вывода инструмента |
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
- В поставляемом `mcp.json` по умолчанию включены (`"enabled": true`):
  - `context7` для документации, setup и API reference.
  - `sequential-thinking` для сложных reasoning-задач.
- Набор активных серверов определяется полем `"enabled"` в `mcp.json`, а не кодом агента — можно менять без правки исходников.

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
- покажет unified overview по модели, backend, session/thread, approvals и MCP;
- выведет краткий cheatsheet по основным slash-командам.

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
- CLI UX: overview, tools/help panels, session-scoped approval selector, stream layout.

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
- При отказе пользователя (`ACCESS_DENIED`) агент не симулирует и не выдумывает результат — явно сообщает об отказе и спрашивает что делать дальше.
- `cli_exec` отключён по умолчанию и считается high-risk инструментом.
- `stop_background_process` не завершает внешние процессы без `ALLOW_EXTERNAL_PROCESS_CONTROL=true`.
- Для production-режима предпочтителен `postgres` backend; для локального CLI — `sqlite`.
