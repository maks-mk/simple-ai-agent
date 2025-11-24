import streamlit as st
import asyncio
import uuid
import time
from typing import Optional

# Проверка зависимостей
try:
    import nest_asyncio
except ImportError:
    st.error("Библиотека nest_asyncio не найдена. Установите: pip install nest_asyncio")
    st.stop()

from langchain_core.messages import HumanMessage
from agent import create_agent_graph, AgentConfig

# ----------------------------
# 1. КОНФИГ UI
# ----------------------------

# Патчим asyncio для работы внутри Streamlit
nest_asyncio.apply()

st.set_page_config(page_title="Smart AI Agent", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    /* ---------------------------------------------------- */
    /* 1. ОБЩИЕ НАСТРОЙКИ UI (Скрытие стандартных элементов) */
    /* ---------------------------------------------------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ------------------------------------------- */
    /* 2. НАСТРОЙКИ ШАПКИ (Header) */
    /* ------------------------------------------- */
    header[data-testid="stHeader"] {
        height: 1.5rem !important;
        min-height: 1.5rem !important;
        padding-top: 0.25rem !important;
        padding-bottom: 0.25rem !important;
        background-color: rgba(0, 0, 0, 0.8) !important;
    }

    /* Уменьшаем верхний отступ основного контейнера */
    .main .block-container {
        padding-top: 1rem !important;
    }

    /* ------------------------------------------- */
    /* 3. СТИЛИ ДЛЯ ЧАТА И КОДА */
    /* ------------------------------------------- */

    /* Стили для сообщений */
    .stChatMessage {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 1rem;
    }

    /* Темный фон для кода в Markdown */
    .stMarkdown code {
        background-color: #262730 !important;
        color: #ffffff !important;
        border-radius: 4px;
        padding: 0.2rem 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 2. ИНИЦИАЛИЗАЦИЯ ГРАФА (Singleton)
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_agent_graph(temperature: float, max_retries: int, retry_delay: int):
    """
    Создает и кэширует граф агента.
    Пересоздается только если меняются аргументы (температура и т.д.).
    """
    
    with st.spinner(f"🔌 Инициализация агента (Temp: {temperature})... Подключение инструментов..."):
        config = AgentConfig(
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        
        # === ИСПРАВЛЕНИЕ #1: Инициализация в текущем цикле ===
        # Используем get_event_loop().run_until_complete() вместо asyncio.run()
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            agent = loop.run_until_complete(create_agent_graph(config))
            return agent, config, loop
        except Exception as e:
            st.error(f"Critical Error during Agent Creation: {e}")
            raise e

# ----------------------------
# 3. УПРАВЛЕНИЕ СЕССИЕЙ
# ----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# 4. SIDEBAR (НАСТРОЙКИ)
# ----------------------------
with st.sidebar:
    st.title("🤖 AI Control Center")
    
    default_cfg = AgentConfig()
    
    st.markdown("### ⚙️ Настройки генерации")
    
    ui_temperature = st.slider(
        "Temperature (Креативность)", 
        min_value=0.0, max_value=1.0, 
        value=default_cfg.temperature, 
        step=0.1,
        help="0 - строгая логика, 1 - творческий полет."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        ui_max_retries = st.number_input(
            "Max Retries", 
            min_value=1, max_value=10, 
            value=default_cfg.max_retries,
            help="Попыток при ошибке API"
        )
    with col2:
        ui_retry_delay = st.number_input(
            "Delay (s)", 
            min_value=0, max_value=10, 
            value=default_cfg.retry_delay,
            help="Пауза между попытками"
        )
    
    st.divider()
    
    # Инфо о провайдере
    model_name = default_cfg.gemini_model if default_cfg.provider == "gemini" else default_cfg.openai_model
    provider_color = "green" if default_cfg.provider == "openai" else "blue"
    st.markdown(f"🧠 Провайдер: **:{provider_color}[{default_cfg.provider.upper()}]**")
    st.caption(f"Модель: `{model_name}`")
    
    st.divider()
    
    col_new, col_info = st.columns([2, 1])
    with col_new:
        if st.button("🔄 Новый чат", type="primary", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
            
    with st.expander("🛠️ Debug Info"):
        st.text(f"Session: {st.session_state.session_id[:8]}")
        st.markdown("""
        **Tools:**
        - 🧠 Memory (ChromaDB)
        - 📂 File System
        - 🔌 MCP Servers
        """)

# === ИНИЦИАЛИЗАЦИЯ ===
try:
    cached_agent, current_config, agent_loop = get_agent_graph(
        ui_temperature,
        ui_max_retries,
        ui_retry_delay,
    )
except Exception:
    st.stop()

# ----------------------------
# 5. ОТРИСОВКА ЧАТА
# ----------------------------
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown(f"### 👋 Привет! Я Smart Agent.\nЯ умею работать с файлами, помнить контекст и использовать внешние инструменты.")
        
    for role, text in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(text)

# ----------------------------
# 6. ЛОГИКА АГЕНТА
# ----------------------------
async def process_stream(user_input: str, status_box):
    """Асинхронный генератор ответа"""
    config = {"configurable": {"thread_id": st.session_state.session_id}}
    text_buffer = ""
    resp_container = st.empty()
    
    try:
        async for event in cached_agent.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages"
        ):
            message, meta = event
            node = meta.get("langgraph_node")
            
            # 1. Обработка текста от LLM
            if node == "agent" and message.content:
                chunk = message.content
                if isinstance(chunk, list):
                    chunk = "".join(p.get("text", "") for p in chunk if isinstance(p, dict))
                
                if isinstance(chunk, str) and chunk:
                    text_buffer += chunk
                    resp_container.markdown(text_buffer + "▌")
                    
            # 2. Обработка вызова инструментов (Tool Calls)
            elif node == "agent" and hasattr(message, "tool_calls") and message.tool_calls:
                if text_buffer.strip():
                    status_box.markdown(f"**💭 Мысль:**\n{text_buffer}")
                    status_box.markdown("---")
                    text_buffer = "" 
                    resp_container.empty()
                
                for tc in message.tool_calls:
                    status_box.write(f"🛠️ **Вызов инструмента:** `{tc['name']}`")
            
            # 3. Обработка результата инструментов
            elif node == "tools":
                tool_name = getattr(message, "name", "Tool")
                content = str(message.content)
                with status_box.expander(f"✅ Результат: {tool_name}", expanded=False):
                    st.code(content[:1500] + ("..." if len(content) > 1500 else ""))

        # Финальный вывод
        resp_container.markdown(text_buffer)
        return text_buffer
        
    except Exception as e:
        # Обработка ошибок внутри стрима
        if current_config.retry_delay > 0:
             status_box.warning(f"Ошибка потока. Пауза {current_config.retry_delay}с...")
             time.sleep(current_config.retry_delay)
        raise e


# ----------------------------
# 7. ОБРАБОТКА ВВОДА
# ----------------------------
if user_input := st.chat_input("Введите запрос..."):
    # 1. Добавляем вопрос пользователя
    st.session_state.messages.append(("user", user_input))
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)
            
    # 2. Запускаем ответ ассистента
    with chat_container:
        with st.chat_message("assistant"):
            status_box = st.status("🧠 Анализирую запрос...", expanded=True)
            
            try:
                # === ИСПРАВЛЕНИЕ #2: Запуск стрима в текущем цикле ===
                # Это гарантирует, что мы используем цикл, в котором был создан агент.
                full_response = agent_loop.run_until_complete(process_stream(user_input, status_box))
                # ======================================================
                
                # Успешное завершение
                status_box.update(label="Готово", state="complete", expanded=False)
                
                # Сохраняем в историю ТОЛЬКО финальный текст (без мыслей)
                if full_response:
                    st.session_state.messages.append(("assistant", full_response))
                    
            except Exception as e:
                status_box.update(label="Произошла ошибка", state="error")
                st.error(f"Ошибка выполнения: {e}")