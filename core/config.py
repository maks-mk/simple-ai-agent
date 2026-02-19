import sys
from pathlib import Path
from typing import Literal, Optional, Any

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.constants import BASE_DIR

class AgentConfig(BaseSettings):
    """
    Конфигурация агента, загружаемая из переменных окружения и .env файла.
    """
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / '.env',
        env_file_encoding='utf-8', 
        extra='ignore'
    )
    
    # Paths
    prompt_path: Path = Field(default=BASE_DIR / "prompt.txt", alias="PROMPT_PATH")
    mcp_config_path: Path = BASE_DIR / "mcp.json"
    
    # Provider Settings
    provider: Literal["gemini", "openai"] = "gemini"
    
    # Tavily Search
    tavily_api_key: Optional[SecretStr] = Field(default=None, alias="TAVILY_API_KEY")

    # Gemini
    gemini_api_key: Optional[SecretStr] = None
    gemini_model: str = "gemini-1.5-flash"
    
    # OpenAI / Compatible
    openai_api_key: Optional[SecretStr] = None
    openai_model: str = "gpt-4o"
    openai_base_url: Optional[str] = None

    # Common Logic
    temperature: float = 0.2
    max_loops: int = Field(default=50, description="Limit steps per request")
    
    # Features Toggle
    enable_search_tools: bool = Field(default=True, alias="ENABLE_SEARCH_TOOLS")
    model_supports_tools: bool = Field(default=True, alias="MODEL_SUPPORTS_TOOLS")
    use_system_tools: bool = Field(default=True, alias="ENABLE_SYSTEM_TOOLS")
    enable_filesystem_tools: bool = Field(default=True, alias="ENABLE_FILESYSTEM_TOOLS")
    enable_process_tools: bool = Field(default=False, alias="ENABLE_PROCESS_TOOLS")
    enable_shell_tool: bool = Field(default=False, alias="ENABLE_SHELL_TOOL")
    
    # Tools Limits
    max_tool_output_length: int = Field(default=4000, alias="MAX_TOOL_OUTPUT")
    max_file_size: int = Field(default=10 * 1024 * 1024, alias="MAX_FILE_SIZE")
    max_background_processes: int = Field(default=5, alias="MAX_BACKGROUND_PROCESSES")
    max_search_chars: int = Field(default=10000, alias="MAX_SEARCH_CHARS")
    max_read_lines: int = Field(default=2000, alias="MAX_READ_LINES")
    
    # Deterministic Mode
    strict_mode: bool = Field(default=False, alias="STRICT_MODE")
    
    # Summarization
    summary_threshold: int = Field(default=20, alias="SESSION_SIZE")
    summary_keep_last: int = Field(default=4, alias="SUMMARY_KEEP_LAST")
    
    # Network / Retry
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    retry_delay: int = Field(default=2, alias="RETRY_DELAY")
    debug: bool = Field(default=False, alias="DEBUG")

    @property
    def safety(self):
        from core.safety_policy import SafetyPolicy
        return SafetyPolicy(
            max_tool_output=self.max_tool_output_length,
            max_file_size=self.max_file_size,
            max_background_processes=self.max_background_processes,
            max_search_chars=self.max_search_chars,
            max_read_lines=self.max_read_lines,
            allow_shell=self.enable_shell_tool
        )

    @model_validator(mode='after')
    def validate_provider_keys(self) -> 'AgentConfig':
        if self.provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY required for gemini provider.")
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for openai provider.")
        return self
