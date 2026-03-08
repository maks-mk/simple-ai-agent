from typing import TypedDict, Annotated, List, NotRequired
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

    # Internal critic state
    critic_status: NotRequired[str]
    critic_source: NotRequired[str]
    critic_feedback: NotRequired[str]
