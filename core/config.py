import functools
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import Field, SecretStr, model_validator, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.constants import BASE_DIR

# --- Defaults ---
DEFAULT_MAX_FILE_SIZE = 300 * 1024 * 1024  # 300 MB
DEFAULT_READ_LIMIT = 2000

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
    mcp_config_path: Path = Field(default=BASE_DIR / "mcp.json", alias="MCP_CONFIG_PATH")
    
    # Provider Settings
    provider: Literal["gemini", "openai"] = Field(default="gemini", alias="PROVIDER")
    
    # Tavily Search
    tavily_api_key: Optional[SecretStr] = Field(default=None, alias="TAVILY_API_KEY")

    # Gemini
    gemini_api_key: Optional[SecretStr] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")
    
    # OpenAI / Compatible
    openai_api_key: Optional[SecretStr] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    openai_base_url: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")

    # Common Logic
    temperature: float = Field(default=0.2, alias="TEMPERATURE")
    max_loops: int = Field(default=50, alias="MAX_LOOPS", description="Limit steps per request")
    
    # Features Toggle
    enable_search_tools: bool = Field(default=True, alias="ENABLE_SEARCH_TOOLS")
    model_supports_tools: bool = Field(default=True, alias="MODEL_SUPPORTS_TOOLS")
    use_system_tools: bool = Field(default=True, alias="ENABLE_SYSTEM_TOOLS")
    enable_filesystem_tools: bool = Field(default=True, alias="ENABLE_FILESYSTEM_TOOLS")
    enable_process_tools: bool = Field(default=False, alias="ENABLE_PROCESS_TOOLS")
    enable_shell_tool: bool = Field(default=False, alias="ENABLE_SHELL_TOOL")
    
    # Tools Limits
    max_tool_output_length: int = Field(default=4000, alias="MAX_TOOL_OUTPUT")
    max_file_size: int = Field(default=DEFAULT_MAX_FILE_SIZE, alias="MAX_FILE_SIZE", description="Max file size in bytes")
    max_background_processes: int = Field(default=5, alias="MAX_BACKGROUND_PROCESSES")
    max_search_chars: int = Field(default=15000, alias="MAX_SEARCH_CHARS")
    max_read_lines: int = Field(default=DEFAULT_READ_LIMIT, alias="MAX_READ_LINES")
    
    # Deterministic Mode
    strict_mode: bool = Field(default=False, alias="STRICT_MODE")
    
    # Summarization
    summary_threshold: int = Field(default=8000, alias="SESSION_SIZE", description="Estimated input context tokens before summarizing (~chars/3)")
    summary_keep_last: int = Field(default=4, alias="SUMMARY_KEEP_LAST")
    
    # Network / Retry
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    retry_delay: int = Field(default=2, alias="RETRY_DELAY")
    debug: bool = Field(default=False, alias="DEBUG")

    @field_validator('max_file_size', mode='before')
    @classmethod
    def parse_max_file_size(cls, v: Union[int, float, str]) -> int:
        """
        Auto-convert MB to bytes if value is small (< 10000).
        Assumes user meant MB if they enter '400' instead of '400000000'.
        """
        try:
            val = float(v)
        except (ValueError, TypeError):
            # Fallback to default if .env contains unparseable string like "300MB"
            return DEFAULT_MAX_FILE_SIZE
            
        # Heuristic: If value is less than 10000, assume it's in MB.
        if val < 10000:
            return int(val * 1024 * 1024)
        return int(val)

    @functools.cached_property
    def safety(self):
        """Returns SafetyPolicy object. Cached to prevent multiple imports and instantiations."""
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
            # Bypass API key check if a base_url is provided (common for local models like Ollama/vLLM)
            if not self.openai_base_url:
                raise ValueError("OPENAI_API_KEY required for openai provider.")
                
        return self