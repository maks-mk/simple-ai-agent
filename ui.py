import streamlit as st
import asyncio
import uuid
import time
from typing import Dict, Any, Optional

# === ПРОВЕРКА ЗАВИСИМОСТЕЙ ===
try:
    import nest_asyncio
except ImportError:
    st.error("Библиотека nest_asyncio не найдена. Установите: pip install nest_asyncio")
    st.stop()

from langchain_core.messages import HumanMessage, BaseMessage
# Импортируем из нашего обновленного agent.py
from agent import create_agent_graph, AgentConfig

# ----------------------------
# 1. КОНФИГУРАЦИЯ СТРАНИЦЫ И СТИЛИ
# ----------------------------

# Патчим asyncio для работы внутри Streamlit
nest_asyncio.apply()

st.set_page_config(page_title="Smart AI Agent", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    /* 1. ГЛАВНОЕ: Тянем контент вверх */
    .block-container {
        padding-top: 1rem !important; /* Было ~6rem, ставим 1.5rem */
        padding-bottom: 2rem !important;
        margin-top: 0 !important;
    }

    /* 2. Сжимаем контейнер шапки */
        header[data-testid="stHeader"] {
        height: 2rem !important;
        min-height: 1.5rem !important;
        padding-top: 0.25rem !important;
        padding-bottom: 0.25rem !important;
        background-color: rgba(0, 0, 0, 0.2) !important;
    }

    /* Уменьшаем верхний отступ основного контейнера */
        .main .block-container {
            padding-top: 1rem !important;
    }
    
    /* 3. Убираем радужную полоску сверху (она занимает место) */
    div[data-testid="stDecoration"] {
        display: none;
    }

    /* 4. Поднимаем кнопки меню (гамбургер и Deploy), чтобы они влезли в узкую шапку */
    div[data-testid="stToolbar"] {
        top: 0rem !important; /* Прижимаем к самому верху */
        right: 2rem !important;
        height: 2.5rem !important;
    }
    
    /* (Опционально) Убираем лишние отступы у кнопок внутри меню */
    div[data-testid="stToolbar"] button {
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 2. ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ
# ----------------------------

class TokenTracker:
    """Класс для подсчета токенов в стриме."""
    def __init__(self):
        self.usage_stats: Dict[str, Dict[str, int]] = {}

    def update(self, message: BaseMessage):
        """Обновляет статистику на основе метаданных сообщения."""
        if not hasattr(message, "usage_metadata") or not message.usage_metadata:
            return

        msg_id = getattr(message, "id", "unknown")
        new_usage = message.usage_metadata

        if msg_id in self.usage_stats:
            current = self.usage_stats[msg_id]
            # Берем MAX, так как в стриме данные могут приходить кумулятивно
            self.usage_stats[msg_id] = {
                "input_tokens": max(current.get("input_tokens", 0), new_usage.get("input_tokens", 0)),
                "output_tokens": max(current.get("output_tokens", 0), new_usage.get("output_tokens", 0)),
            }
        else:
            self.usage_stats[msg_id] = new_usage

    def get_totals(self) -> Dict[str, int]:
        total_in = sum(s.get("input_tokens", 0) for s in self.usage_stats.values())
        total_out = sum(s.get("output_tokens", 0) for s in self.usage_stats.values())
        return {"in": total_in, "out": total_out, "total": total_in + total_out}

    def get_display_html(self) -> str:
        stats = self.get_totals()
        if stats['total'] == 0:
            return ""
        return f"""
        <div class='token-badge'>
            🪙 Tokens: <b>{stats['total']}</b> (In: {stats['in']} / Out: {stats['out']})
        </div>
        """

# ----------------------------
# 3. ИНИЦИАЛИЗАЦИЯ АГЕНТА (Cached)
# ----------------------------
@st.cache_resource(show_spinner=False)
def init_agent_system(
    provider: str,
    model: str, 
    temp: float, 
    max_retries: int
):
    """
    Создает и кэширует граф агента. 
    Используем параметры примитивных типов для корректного хеширования кэша.
    """
    with st.spinner("🔌 Подключение нейронных сетей..."):
        # 1. Загружаем базу из ENV
        config = AgentConfig.from_env()
        
        # 2. Применяем оверрайды из UI
        config.provider = provider
        config.temperature = temp
        config.max_retries = max_retries
        
        if provider == "gemini":
            config.gemini_model = model
        else:
            config.openai_model = model

        # 3. Создаем цикл и агента
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            agent = loop.run_until_complete(create_agent_graph(config))
            return agent, config, loop
        except Exception as e:
            st.error(f"Critical Error: {e}")
            raise e

# ----------------------------
# 4. SIDEBAR И НАСТРОЙКИ
# ----------------------------
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    # Загружаем дефолты, чтобы показать в UI
    env_config = AgentConfig.from_env()
    
    st.subheader("Model Config")
    
    # Выбор провайдера (если ключи есть)
    provider_options = []
    if env_config.gemini_key: provider_options.append("gemini")
    if env_config.openai_key: provider_options.append("openai")
    
    if not provider_options:
        st.error("Нет API ключей в .env!")
        st.stop()
        
    selected_provider = st.selectbox("Provider", provider_options, index=provider_options.index(env_config.provider) if env_config.provider in provider_options else 0)
    
    # Модель (просто текстовое поле для гибкости)
    default_model = env_config.gemini_model if selected_provider == "gemini" else env_config.openai_model
    selected_model = st.text_input("Model Name", value=default_model)
    
    st.subheader("Generation")
    ui_temperature = st.slider("Temperature", 0.0, 1.0, env_config.temperature, 0.1)
    ui_max_retries = st.slider("Max Retries", 1, 5, env_config.max_retries)
    
    st.divider()
    
    if st.button("🗑️ Очистить историю", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    with st.expander("Session Info"):
        st.caption(f"ID: {st.session_state.get('session_id', 'init')}")

# Инициализация (Singleton)
try:
    agent, config, agent_loop = init_agent_system(
        selected_provider, 
        selected_model, 
        ui_temperature, 
        ui_max_retries
    )
except Exception:
    st.stop()

# ----------------------------
# 5. УПРАВЛЕНИЕ СОСТОЯНИЕМ ЧАТА
# ----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# 6. ОТРИСОВКА ИСТОРИИ
# ----------------------------
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.info("👋 Привет! Я готов к работе. Задай мне вопрос или попроси выполнить задачу.")
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Отрисовка статистики токенов, если она есть в истории
            if "tokens_html" in msg:
                st.markdown(msg["tokens_html"], unsafe_allow_html=True)

# ----------------------------
# 7. ЛОГИКА ОБРАБОТКИ (STREAM)
# ----------------------------
async def run_agent_stream(user_input: str, status_placeholder):
    """Асинхронный запуск агента с обновлением UI."""
    
    cfg = {"configurable": {"thread_id": st.session_state.session_id}}
    
    full_text = ""
    token_tracker = TokenTracker()
    response_placeholder = st.empty()
    
    # Таймер
    start_time = time.time()

    try:
        # Запуск стрима
        async for event in agent.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config=cfg,
            stream_mode="messages"
        ):
            message, meta = event
            node = meta.get("langgraph_node")
            
            # 1. Считаем токены
            token_tracker.update(message)
            
            # 2. Обработка ответа агента (LLM)
            if node == "agent":
                # Обработка вызова инструментов
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tc in message.tool_calls:
                        status_placeholder.write(f"🛠️ **Использую инструмент:** `{tc['name']}`")
                
                # Обработка текстового контента
                elif message.content:
                    chunk = message.content
                    if isinstance(chunk, list):
                         # Мультимодальный контент -> текст
                        chunk = "".join(x["text"] for x in chunk if "text" in x)
                    
                    if chunk:
                        full_text += chunk
                        response_placeholder.markdown(full_text + "▌")

            # 3. Обработка результата инструмента
            elif node == "tools":
                tool_name = getattr(message, "name", "tool")
                content_preview = str(message.content)[:500]
                if len(str(message.content)) > 500: content_preview += "..."
                
                with status_placeholder.expander(f"✅ Результат: {tool_name}", expanded=False):
                    st.code(content_preview)

        # Финализация
        response_placeholder.markdown(full_text)
        duration = time.time() - start_time
        
        return full_text, token_tracker.get_display_html()

    except Exception as e:
        # Логируем, но не крашим весь UI
        st.error(f"Ошибка потока: {e}")
        return full_text, ""

# ----------------------------
# 8. ОБРАБОТКА ВВОДА ПОЛЬЗОВАТЕЛЯ
# ----------------------------
if prompt := st.chat_input("Введите сообщение..."):
    # 1. Сохраняем и показываем вопрос
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    # 2. Генерируем ответ
    with chat_container:
        with st.chat_message("assistant"):
            # Контейнер для статуса (мысли, инструменты)
            status_box = st.status("🧠 Думаю...", expanded=True)
            
            # Запуск внутри сохраненного цикла событий
            response_text, token_html = agent_loop.run_until_complete(
                run_agent_stream(prompt, status_box)
            )
            
            # Завершение статуса
            status_box.update(label="Готово", state="complete", expanded=False)
            
            # Если ответ пустой (ошибка), не добавляем в историю
            if response_text:
                st.markdown(token_html, unsafe_allow_html=True)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "tokens_html": token_html
                })