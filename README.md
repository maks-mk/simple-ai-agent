# Autonomous AI Agent

**v0.47b**

Автономный CLI-агент на базе LangGraph с локальными инструментами, MCP-интеграцией, внутренним critic-узлом и более строгими safety-контрактами для файлов, tool-calling и фоновых процессов.

---

## Что изменилось в `0.47b`

- Обновлена структура `core/`: pure text/helpers вынесены из CLI-зависимого слоя.
- Ошибочные `tool_calls` больше не превращаются в синтетический `unknown_tool`, а обрабатываются как protocol error.
- Reflection/retry больше не ломают порядок ролей в истории сообщений после `tool`.
- `MAX_FILE_SIZE` теперь парсится строго: число без суффикса трактуется как байты, для крупных лимитов нужны явные единицы (`300MiB`, `4MB`).
- `run_background_process` ужесточен: запрещены shell-операторы, рабочая директория валидируется внутри workspace.

---

## Возможности

### Интерфейс
- Потоковый вывод ответов и статусов инструментов.
- CLI на `rich` + `prompt_toolkit`.
- Fuzzy autocomplete для команд и путей.
- Восстановление сессии после прерванных запусков.

### Граф агента
- Основной поток: `summarize -> update_step -> agent -> tools/critic`.
- Внутренний `critic` не дает завершить задачу, если ответ формально есть, а действие не доведено до конца.
- Автосжатие контекста по `SESSION_SIZE` с сохранением последних сообщений.
- Loop guard на уровне графа и на уровне повторяющихся tool-вызовов.

### Инструменты
- Filesystem tools для чтения, записи, редактирования и поиска по проекту.
- Search tools через Tavily.
- System tools для системной и сетевой диагностики.
- Process tools для фоновых процессов с дополнительными ограничениями.
- Опциональный shell tool.
- MCP tools, подключаемые из `mcp.json`.

---

## Текущая структура проекта

```text
.
├── core/
│   ├── cli_utils.py
│   ├── config.py
│   ├── constants.py
│   ├── errors.py
│   ├── fuzzy_completer.py
│   ├── logging_config.py
│   ├── message_utils.py
│   ├── nodes.py
│   ├── safety_policy.py
│   ├── session_utils.py
│   ├── state.py
│   ├── stream_processor.py
│   ├── text_utils.py
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
- `config.py` загружает `.env`, парсит лимиты и флаги, включая строгий `MAX_FILE_SIZE`.
- `nodes.py` содержит LangGraph-узлы, retry/reflection-логику и обработку protocol errors.
- `state.py` описывает состояние графа.
- `message_utils.py` и `text_utils.py` содержат dependency-free helper-функции для текста и сообщений.
- `cli_utils.py`, `ui_theme.py`, `fuzzy_completer.py`, `stream_processor.py` отвечают за CLI-слой.
- `session_utils.py`, `validation.py`, `errors.py`, `safety_policy.py` отвечают за ремонт сессии, валидацию, типизацию ошибок и ограничения безопасности.

### `tools/`
- `filesystem.py` реализует чтение/запись/редактирование/поиск по файлам с path guards и аккуратным repair для очевидных опечаток в путях.
- `search_tools.py` подключает Tavily search/fetch API.
- `system_tools.py` отдает системную и сетевую информацию.
- `process_tools.py` управляет фоновыми процессами без `shell=True` и с проверкой `cwd`.
- `local_shell.py` содержит опциональный shell tool.
- `tool_registry.py` собирает локальные и MCP-инструменты в единый registry.

---

## Конфигурация `.env`

### Провайдер модели

| Параметр | Назначение |
| :--- | :--- |
| `PROVIDER` | `gemini` или `openai` |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` | ключ выбранного провайдера |
| `GEMINI_MODEL` / `OPENAI_MODEL` | имя модели |
| `OPENAI_BASE_URL` | base URL для OpenAI-совместимых API |

### Поведение агента

| Параметр | Значение по умолчанию |
| :--- | :--- |
| `TEMPERATURE` | `0.2` |
| `MAX_LOOPS` | `50` |
| `PROMPT_PATH` | `prompt.txt` |
| `MODEL_SUPPORTS_TOOLS` | `true` |
| `DEBUG` | `false` |

### Включение подсистем

| Параметр | Значение по умолчанию |
| :--- | :--- |
| `ENABLE_FILESYSTEM_TOOLS` | `true` |
| `ENABLE_SEARCH_TOOLS` | `true` |
| `ENABLE_SYSTEM_TOOLS` | `true` |
| `ENABLE_PROCESS_TOOLS` | `false` |
| `ENABLE_SHELL_TOOL` | `false` |

### Лимиты и безопасность

| Параметр | Значение по умолчанию | Комментарий |
| :--- | :--- | :--- |
| `MAX_TOOL_OUTPUT` | `10000` | лимит вывода инструмента |
| `MAX_SEARCH_CHARS` | `15000` | лимит search/fetch контента |
| `MAX_FILE_SIZE` | `300MiB` | число без суффикса = байты |
| `MAX_READ_LINES` | `2000` | лимит строк при чтении |
| `MAX_BACKGROUND_PROCESSES` | `5` | лимит фоновых задач |

Для `MAX_FILE_SIZE` используйте явные единицы: `4MB`, `300MiB`, `1GiB`. Значение `300` теперь означает именно `300 bytes`, а не `300 MB`.

---

## MCP

- Конфигурация серверов находится в `mcp.json`.
- Подключаются только записи с `"enabled": true`.
- MCP tools регистрируются вместе с локальными инструментами и доступны агенту наравне с ними.

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
- включите нужные tool-подсистемы;
- при необходимости активируйте MCP-серверы в `mcp.json`.

### 4. Запуск

```bash
python agent_cli.py
```

---

## Тесты

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

---

## Команды в CLI

- `/help` — краткая справка
- `/tools` — список доступных инструментов
- `clear` / `reset` — сброс текущей сессии
- `exit` / `quit` — выход
- `Alt+Enter` — многострочный ввод
