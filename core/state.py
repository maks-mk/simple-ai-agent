from typing import TypedDict, Annotated, List, NotRequired, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Simplified Agent State.
    """
    # Message history
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Compressed memory
    summary: str
    
    # Step counter
    steps: int
    
    # Token usage tracking (Last step usage)
    token_usage: dict

    # Original user task for the current request
    current_task: NotRequired[str]

    # Deterministic turn-control state
    retry_count: NotRequired[int]
    retry_reason: NotRequired[str]
    turn_outcome: NotRequired[str]
    final_issue: NotRequired[str]

    # Durable runtime/session info
    session_id: NotRequired[str]
    run_id: NotRequired[str]
    turn_id: NotRequired[int]
    pending_approval: NotRequired[Dict[str, Any] | None]
    open_tool_issue: NotRequired[Dict[str, Any] | None]
    last_tool_error: NotRequired[str]
    last_tool_result: NotRequired[str]
    safety_mode: NotRequired[str]
