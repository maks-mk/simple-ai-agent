# Autonomous CLI AI Agent

Локальный CLI-агент на базе LangGraph с детерминированным графом, approval-интерраптами для опасных действий, MCP-интеграцией и долговечным checkpoint/session state.

## Ключевые особенности

- Граф без `critic`-узла: контроль хода делается через явные поля состояния (`retry_count`, `retry_reason`, `turn_outcome`, `final_issue`).
- Маршрутизация после `agent` детерминированная: `approval | tools | prepare_retry | finalize_blocked | END`.
- Retry/recovery-бюджет ограничен и настраивается через `MAX_RECOVERY_ATTEMPTS`; по умолчанию агент допускает несколько LLM-recoverable попыток в одном ходе, а не только один авто-повтор.
- После реального прогресса recovery-счётчик сбрасывается, чтобы успешные промежуточные шаги не “съедали” шанс на дальнейшую нормальную работу.
- Approval flow построен через LangGraph `interrupt`, с решениями `Yes / No / Always` в CLI.
- Tool metadata задаётся явно для локальных tools; для MCP применяется безопасный дефолт + минимальная эвристика.
- Для MCP-tools по умолчанию включён approval; авто-исключение только для имён с `read`, `search`, `find`.
- Добавлены минимальные guardrails от рекурсивно-разрушительных операций по защищённым корням (в filesystem и shell-пути).

## Архитектура

### Основные точки

- [agent.py](D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/agent.py) — сборка LLM, ToolRegistry, checkpoint runtime и компиляция графа.
- [agent_cli.py](D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/agent_cli.py) — локальный UI, slash-команды, approval UX, запуск потокового рендера.

### Граф выполнения

```text
START
  -> summarize
  -> update_step
  -> agent
      -> approval (если tool calls требуют approval)
      -> tools (если tool calls не требуют approval)
      -> prepare_retry (если нужен повтор и retry_count < 1)
      -> finalize_blocked (если повтор исчерпан / loop guard)
      -> END (если финальный ответ валиден)

approval -> tools
tools -> update_step | prepare_retry | finalize_blocked
prepare_retry -> update_step
finalize_blocked -> END
```

### Состояние графа

Ключевые поля в [core/state.py](D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/core/state.py):

- `retry_count`
- `retry_reason`
- `turn_outcome`
- `final_issue`
- `pending_approval`
- `open_tool_issue`
- `last_tool_error`
- `last_tool_result`

## CLI / UX

Интерфейс:

- Компактная стартовая overview-панель (provider/model/tools/backend/approvals/MCP/status).
- Потоковый вывод tool-вызовов в формате:
  - `tool > <tool_name>(args)`
  - `             └→ <result/error> <duration>`
- Фазы рантайма: `Thinking`, `Running tools`, `Awaiting approval`, `Recovering`, `Finishing`.
- При tool-ошибках граф не останавливается сразу: если ошибка выглядит recoverable, агент получает повторный шанс перепланировать ход; если бюджет исчерпан, ход завершается контролируемо через `finalize_blocked`.
- Approval-панель в минимальном виде (`Action`, `Target`) без лишних метаданных.
- Выбор approval через стрелки `↑/↓`, подтверждение `Enter`, отмена `Esc`.
- Нижний toolbar в едином спокойном стиле, без визуального шума.

Slash-команды:

- `/help`
- `/tools`
- `/session`
- `/new`
- `/quit`
- `clear` / `reset`

## Инструменты и безопасность

### Реестр инструментов

[tools/tool_registry.py](D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/tools/tool_registry.py):

- Локальные tools (filesystem/search/system/process/shell).
- MCP tools из `mcp.json`.
- Единая metadata-политика на tool-уровне.

### ToolMetadata

[core/tool_policy.py](D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/core/tool_policy.py):

- `read_only`
- `mutating`
- `destructive`
- `requires_approval`
- `networked`
- `impact_scope`
- `ui_kind`
- `source`

### Guardrails

- [core/destructive_guardrails.py](D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/core/destructive_guardrails.py)
- [tools/filesystem_impl/pathing.py](D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/tools/filesystem_impl/pathing.py)
- [tools/local_shell.py](D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/tools/local_shell.py)

Блокируются очевидные рекурсивно-разрушительные операции по защищённым root/system-путям.

## Конфигурация

Основная конфигурация в [core/config.py](D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/core/config.py).

Ключевые env-переменные:

- `PROVIDER`, `OPENAI_API_KEY` / `GEMINI_API_KEY`, `OPENAI_MODEL` / `GEMINI_MODEL`
- `OPENAI_BASE_URL`
- `PROMPT_PATH`, `MCP_CONFIG_PATH`
- `CHECKPOINT_BACKEND` (`sqlite|memory|postgres`)
- `CHECKPOINT_SQLITE_PATH`, `CHECKPOINT_POSTGRES_URL`
- `SESSION_STATE_PATH`, `RUN_LOG_DIR`
- `ENABLE_FILESYSTEM_TOOLS`, `ENABLE_SEARCH_TOOLS`, `ENABLE_SYSTEM_TOOLS`, `ENABLE_PROCESS_TOOLS`, `ENABLE_SHELL_TOOL`, `ENABLE_APPROVALS`
- `ALLOW_EXTERNAL_PROCESS_CONTROL`
- `MAX_LOOPS`, `MAX_RETRIES`, `RETRY_DELAY`
- `MAX_RECOVERY_ATTEMPTS`
- `SESSION_SIZE`, `SUMMARY_KEEP_LAST`
- `MAX_TOOL_OUTPUT`, `MAX_FILE_SIZE`, `MAX_READ_LINES`, `MAX_SEARCH_CHARS`, `MAX_BACKGROUND_PROCESSES`

### Recovery policy

- `MAX_RETRIES` относится к низкоуровневым попыткам вызова модели.
- `MAX_RECOVERY_ATTEMPTS` ограничивает LLM-recoverable перепланирование внутри одного user turn.
- Любой успешный tool-result или новый валидный tool-call сбрасывает recovery streak.

## MCP

- Конфиг серверов: [mcp.json](D:/py_projects/simple_ai_agent/agent+stategraph/v7.3b/mcp.json)
- Подключаются только записи с `"enabled": true`.
- `tool_metadata` overrides опциональны; базовый режим работает без ручного описания каждого tool.
- Если override не задан, для MCP действует дефолтная политика (approval required, кроме эвристики `read/search/find`).

## Установка и запуск

### 1. Окружение

```bash
python -m venv venv
```

Windows:

```powershell
.\venv\Scripts\activate
```

### 2. Зависимости

```bash
pip install -r requirements.txt
```

### 3. Настройка `.env`

```powershell
Copy-Item env_example.txt .env
```

Заполните ключи провайдера и нужные feature flags.

### 4. Запуск

```bash
python agent_cli.py
```

## Тесты

Базовый прогон:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

Часто используемые наборы:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_cli_ux.py"
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_runtime_refactor.py"
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_critic_graph.py"
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_tooling_refactor.py"
```

Примечание: `test_pdf_mcp.py` может зависеть от локальных прав/окружения (временные директории, reportlab/fitz).

## Структура проекта (кратко)

```text
core/         # граф, state, policy, stream/UI processor, config, logging
tools/        # локальные инструменты и реестр
tests/        # unit/integration тесты
pdf_mcp/      # отдельный MCP-сервер для PDF
agent.py      # сборка runtime + графа
agent_cli.py  # локальный CLI
mcp.json      # MCP-конфигурация
prompt.txt    # базовый системный prompt
```
