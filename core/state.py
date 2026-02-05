from typing import TypedDict, Annotated, List
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
    
    # Token usage tracking
    token_used: int
