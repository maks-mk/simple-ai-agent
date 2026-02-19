from pydantic import BaseModel, Field

class SafetyPolicy(BaseModel):
    """
    Centralized configuration for safety limits and policies.
    """
    max_tool_output: int = Field(default=5000, description="Maximum characters for tool output")
    max_file_size: int = Field(default=10 * 1024 * 1024, description="Maximum file size in bytes (10MB)")
    max_background_processes: int = Field(default=5, description="Maximum concurrent background processes")
    max_search_chars: int = Field(default=10000, description="Maximum characters for search results")
    max_read_lines: int = Field(default=2000, description="Maximum lines to read from a file")
    allow_shell: bool = Field(default=False, description="Allow execution of shell commands")
