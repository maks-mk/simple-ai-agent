from typing import Any
from rich.console import Console
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

def repair_session_if_needed(agent_app: Any, thread_id: str, console: Console):
    """
    Checks for interrupted tool calls in the session history and repairs them
    by inserting error messages for uncompleted calls.
    This prevents the agent from getting stuck or hallucinating tool outputs.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        current_state = agent_app.get_state(config)
        
        if not current_state or not current_state.values:
            return

        messages = current_state.values.get("messages", [])
        if not messages:
            return

        # Find the last AIMessage with tool calls
        last_ai_msg = None
        last_ai_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if isinstance(m, (AIMessage, AIMessageChunk)) and m.tool_calls:
                last_ai_msg = m
                last_ai_idx = i
                break
        
        if last_ai_msg:
            # Gather existing ToolMessages after this AIMessage
            existing_tool_outputs = set()
            for j in range(last_ai_idx + 1, len(messages)):
                m = messages[j]
                if isinstance(m, ToolMessage):
                    existing_tool_outputs.add(m.tool_call_id)
            
            # Identify missing responses
            missing_tool_calls = []
            for tc in last_ai_msg.tool_calls:
                if tc["id"] not in existing_tool_outputs:
                    missing_tool_calls.append(tc)
            
            if missing_tool_calls:
                console.print(f"[dim]⚠ Detected {len(missing_tool_calls)} interrupted tool execution(s). Filling gaps...[/]")
                tool_msgs = []
                for tc in missing_tool_calls:
                    tool_msgs.append(ToolMessage(
                        tool_call_id=tc["id"],
                        content="Error: Execution interrupted (system limit reached or user stop). Please retry.",
                        name=tc["name"]
                    ))
                
                # update_state returns a dict, not awaitable in some versions
                agent_app.update_state(config, {"messages": tool_msgs}, as_node="tools")
                console.print("[dim]✔ History repaired (filled gaps). Ready for new input.[/]")
                
    except Exception as e:
        # Silently fail or log debug if state repair fails, to not block the user
        pass
