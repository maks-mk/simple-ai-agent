import streamlit as st
import asyncio
import uuid
import time # <--- NEW: Нужен для sleep в retry_delay
from langchain_core.messages import HumanMessage
from agent import create_agent_graph, AgentConfig

# ----------------------------
# 1. КОНФИГ UI
# ----------------------------
st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stChatMessage {border: 1px solid #333; border-radius: 10px; padding: 1rem;}
    
    /* ИСПРАВЛЕНИЕ: Темный фон для кода, чтобы сочетался с темой */
    .stMarkdown code {
        background-color: #262730 !important; /* Темно-серый фон */
        color: #ffffff !important;           /* Белый текст */
        border-radius: 4px;
        padding: 0.2rem 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 2. "ВЕЧНАЯ" ИНИЦИАЛИЗАЦИЯ (Singleton)
# ----------------------------
# Теперь функция зависит от параметров. Если они меняются — агент пересоздается.
@st.cache_resource
def get_agent_bundle(temperature, max_retries, retry_delay):
    """
    Создает агента с динамическими настройками.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # === NEW: Передаем параметры из UI в конфиг ===
    # Мы создаем конфиг, переопределяя значения из .env значениями из UI
    config = AgentConfig(
        temperature=temperature,
        max_retries=max_retries,
        retry_delay=retry_delay
    )
    # ==============================================
    
    agent = loop.run_until_complete(create_agent_graph(config))
    
    print(f"✅ SYSTEM: Агент инициализирован (Temp: {temperature})")
    return loop, agent, config

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
    st.title("🤖 AI Control")
    
    # Загружаем дефолтные значения из .env (через пустой конфиг)
    default_cfg = AgentConfig()
    
    st.markdown("### ⚙️ Параметры модели")
    
    # === NEW: Виджеты управления ===
    ui_temperature = st.slider(
        "Temperature (Креативность)", 
        min_value=0.0, max_value=1.0, 
        value=default_cfg.temperature, 
        step=0.1
    )
    
    col1, col2 = st.columns(2)
    with col1:
        ui_max_retries = st.number_input(
            "Max Retries", 
            min_value=1, max_value=10, 
            value=default_cfg.max_retries
        )
    with col2:
        ui_retry_delay = st.number_input(
            "Delay (sec)", 
            min_value=0, max_value=10, 
            value=default_cfg.retry_delay
        )
    # ================================

    st.divider()
    
    model_name = default_cfg.gemini_model if default_cfg.provider == "gemini" else default_cfg.openai_model
    st.markdown(f"🚀 **{default_cfg.provider.upper()}** / **{model_name}**")
    st.info(f"ID сессии: {st.session_state.session_id[:8]}...")
    
    if st.button("🗑️ Новый чат (Сброс контекста)", type="primary"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
        
    with st.expander("ℹ️ Инструменты"):
        st.markdown("""
        **Доступные инструменты:**
        - 📂 File System (Safe Delete)
        - 🔌 MCP Servers (из mcp.json)
        """)

# === ИНИЦИАЛИЗАЦИЯ С ПАРАМЕТРАМИ ===
try:
    # Передаем значения из слайдеров в функцию кэширования
    cached_loop, cached_agent, current_config = get_agent_bundle(
        ui_temperature, 
        ui_max_retries, 
        ui_retry_delay
    )
except Exception as e:
    st.error(f"Ошибка инициализации агента: {e}")
    st.stop()
# ===================================

# ----------------------------
# 5. ОТРИСОВКА ЧАТА
# ----------------------------
chat_box = st.container()
with chat_box:
    if not st.session_state.messages:
        st.markdown("👋 Привет! Я готов помочь с любыми задачами.")
        
    for role, text in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(text)

# ----------------------------
# 6. ОБРАБОТКА ВВОДА
# ----------------------------
user_input = st.chat_input("Отправьте сообщение агенту...")

if user_input:
    st.session_state.messages.append(("user", user_input))
    with chat_box:
        with st.chat_message("user"):
            st.markdown(user_input)
            
    with chat_box:
        with st.chat_message("assistant"):
            resp_container = st.empty()
            status_box = st.status("🤔 Думаю...", expanded=True)
            
            async def process_stream():
                config = {"configurable": {"thread_id": st.session_state.session_id}}
                text_buffer = ""
                
                try:
                    async for event in cached_agent.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=config,
                        stream_mode="messages"
                    ):
                        message, meta = event
                        node = meta.get("langgraph_node")
                        
                        if node == "agent" and message.content:
                            chunk = message.content
                            if isinstance(chunk, list):
                                chunk = "".join(p.get("text", "") for p in chunk if isinstance(p, dict))
                            
                            if isinstance(chunk, str) and chunk:
                                text_buffer += chunk
                                resp_container.markdown(text_buffer + "▌")
                                
                        elif node == "agent" and hasattr(message, "tool_calls") and message.tool_calls:
                            if text_buffer.strip():
                                status_box.markdown(f"**💭 Мысль:**\n{text_buffer}")
                                status_box.markdown("---")
                                text_buffer = "" 
                                resp_container.empty()
                            
                            for tc in message.tool_calls:
                                status_box.write(f"🛠️ **Вызов:** `{tc['name']}`")
                        
                        elif node == "tools":
                            tool_name = getattr(message, "name", "Unknown Tool")
                            content = str(message.content)
                            with status_box.expander(f"✅ Результат: {tool_name}", expanded=False):
                                st.code(content[:1000] + ("..." if len(content) > 1000 else ""))

                    resp_container.markdown(text_buffer)
                    status_box.update(label="Готово", state="complete", expanded=False)
                    return text_buffer
                    
                except Exception as e:
                    status_box.update(label="Ошибка!", state="error")
                    st.error(f"Stream Error: {e}")
                    
                    # === NEW: Использование Delay ===
                    if current_config.retry_delay > 0:
                        st.warning(f"Ожидание {current_config.retry_delay} сек. перед разблокировкой...")
                        time.sleep(current_config.retry_delay)
                    # ================================
                    
                    return f"Error: {e}"

            try:
                final_res = cached_loop.run_until_complete(process_stream())
                if final_res:
                    st.session_state.messages.append(("assistant", final_res))
            except RuntimeError as e:
                st.error(f"Critical Loop Error: {e}")
                if st.button("Перезагрузить страницу"):
                    st.rerun()