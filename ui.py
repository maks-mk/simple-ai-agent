import streamlit as st
import asyncio
import uuid
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
    .stChatMessage {border: 1px solid #eee; border-radius: 10px; padding: 1rem;}
    .stMarkdown code {background-color: #f0f2f6 !important;}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 2. "ВЕЧНАЯ" ИНИЦИАЛИЗАЦИЯ (Singleton)
# ----------------------------
@st.cache_resource
def get_agent_bundle():
    """
    Создает агента и привязанный к нему Event Loop.
    Возвращает кортеж (loop, agent).
    """
    # Создаем новый цикл событий
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Инициализируем агента с дефолтным конфигом
    # (можно расширить, добавив настройки в UI)
    config = AgentConfig()
    agent = loop.run_until_complete(create_agent_graph(config))
    
    print("✅ SYSTEM: Агент инициализирован и закэширован")
    return loop, agent

try:
    cached_loop, cached_agent = get_agent_bundle()
except Exception as e:
    st.error(f"Ошибка инициализации агента: {e}")
    st.stop()

# ----------------------------
# 3. УПРАВЛЕНИЕ СЕССИЕЙ
# ----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# 4. SIDEBAR
# ----------------------------
with st.sidebar:
    st.title("🤖 AI Control")
    
    # Отображаем текущую конфигурацию
    cfg = AgentConfig()
    model_name = cfg.gemini_model if cfg.provider == "gemini" else cfg.openai_model
    #st.caption(f"🚀 **{cfg.provider.upper()}** / `{model_name}`")
    # Используем st.markdown вместо caption для яркости, и убираем ` `
    st.markdown(f"🚀 **{cfg.provider.upper()}** / **{model_name}**")
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

# ----------------------------
# 5. ОТРИСОВКА ЧАТА
# ----------------------------
chat_box = st.container()
with chat_box:
    if not st.session_state.messages:
        st.markdown("👋 Привет! Я готов помочь с кодом и файлами.")
        
    for role, text in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(text)

# ----------------------------
# 6. ОБРАБОТКА ВВОДА
# ----------------------------
user_input = st.chat_input("Отправьте сообщение агенту...")

if user_input:
    # 1. Отображаем сообщение пользователя
    st.session_state.messages.append(("user", user_input))
    with chat_box:
        with st.chat_message("user"):
            st.markdown(user_input)
            
    # 2. Генерация ответа
    with chat_box:
        with st.chat_message("assistant"):
            resp_container = st.empty()
            status_box = st.status("🤔 Думаю...", expanded=True)
            
            async def process_stream():
                config = {"configurable": {"thread_id": st.session_state.session_id}}
                text_buffer = ""
                
                try:
                    # Запускаем стриминг
                    async for event in cached_agent.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=config,
                        stream_mode="messages"
                    ):
                        message, meta = event
                        node = meta.get("langgraph_node")
                        
                        # --- ТЕКСТ ОТ АГЕНТА ---
                        if node == "agent" and message.content:
                            chunk = message.content
                            # Обработка странных форматов LangChain (иногда список)
                            if isinstance(chunk, list):
                                chunk = "".join(p.get("text", "") for p in chunk if isinstance(p, dict))
                            
                            if isinstance(chunk, str) and chunk:
                                text_buffer += chunk
                                resp_container.markdown(text_buffer + "▌")
                                
                        # --- ВЫЗОВ ИНСТРУМЕНТОВ ---
                        elif node == "agent" and hasattr(message, "tool_calls") and message.tool_calls:
                            # Если накопился текст мысли перед инструментом, покажем его в статусе
                            if text_buffer.strip():
                                status_box.markdown(f"**💭 Мысль:**\n{text_buffer}")
                                status_box.markdown("---")
                                text_buffer = "" # Сброс буфера, так как мысль ушла в статус
                                resp_container.empty() # Очищаем основное поле, ждем результат
                            
                            for tc in message.tool_calls:
                                status_box.write(f"🛠️ **Вызов:** `{tc['name']}`")
                        
                        # --- РЕЗУЛЬТАТ ИНСТРУМЕНТОВ ---
                        elif node == "tools":
                            tool_name = getattr(message, "name", "Unknown Tool")
                            content = str(message.content)
                            
                            # Красивый вывод результата
                            with status_box.expander(f"✅ Результат: {tool_name}", expanded=False):
                                st.code(content[:1000] + ("..." if len(content) > 1000 else ""))

                    # Финальное обновление
                    resp_container.markdown(text_buffer)
                    status_box.update(label="Готово", state="complete", expanded=False)
                    return text_buffer
                    
                except Exception as e:
                    status_box.update(label="Ошибка!", state="error")
                    st.error(f"Stream Error: {e}")
                    return f"Error: {e}"

            # Запускаем асинхронную функцию в существующем цикле
            try:
                final_res = cached_loop.run_until_complete(process_stream())
                if final_res:
                    st.session_state.messages.append(("assistant", final_res))
            except RuntimeError as e:
                st.error(f"Critical Loop Error: {e}")
                if st.button("Перезагрузить страницу"):
                    st.rerun()
