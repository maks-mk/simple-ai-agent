import sys
from pathlib import Path
from typing import Literal, Optional, Any

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

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
    memory_db_path: str = str(BASE_DIR / "memory_db")
    
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
    max_loops: int = Field(default=15, description="Limit steps per request")
    token_budget: int = Field(default=30000, alias="TOKEN_BUDGET")
    
    # Features Toggle
    enable_deep_search: bool = Field(default=False, alias="DEEP_SEARCH")
    enable_search_tools: bool = Field(default=True, alias="ENABLE_SEARCH_TOOLS")
    model_supports_tools: bool = Field(default=True, alias="MODEL_SUPPORTS_TOOLS")
    use_long_term_memory: bool = Field(default=False, alias="LONG_TERM_MEMORY")
    use_system_tools: bool = Field(default=True, alias="ENABLE_SYSTEM_TOOLS")
    enable_media_tools: bool = Field(default=False, alias="ENABLE_MEDIA_TOOLS")
    safety_guard_enabled: bool = Field(default=True, alias="SAFETY_GUARD_ENABLED")
    enable_tool_filtering: bool = Field(default=True, alias="ENABLE_TOOL_FILTERING")
    
    # Summarization
    summary_threshold: int = Field(default=20, alias="SESSION_SIZE")
    summary_keep_last: int = Field(default=4, alias="SUMMARY_KEEP_LAST")
    
    # Network / Retry
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    retry_delay: int = Field(default=2, alias="RETRY_DELAY")
    debug: bool = Field(default=False, alias="DEBUG")

    @model_validator(mode='after')
    def validate_provider_keys(self) -> 'AgentConfig':
        if self.provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY required for gemini provider.")
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for openai provider.")
        return self

    def check_tool_support(self) -> bool:
        """
        Определяет, поддерживает ли текущая модель вызов инструментов.
        """
        if not self.model_supports_tools:
            return False
            
        if self.provider == "openai":
            # Эвристика для моделей, которые часто не поддерживают тулы
            model_name = self.openai_model.lower()
            no_tool_prefixes = (
                "tngtech/", "huggingface/", "grey-wing/", "sao10k/" 
            )
            if model_name.startswith(no_tool_prefixes):
                return False
        return True

    def get_llm(self) -> BaseChatModel:
        """
        Инициализирует и возвращает экземпляр LLM на основе настроек.
        """
        if self.provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=self.gemini_model,
                temperature=self.temperature,
                google_api_key=self.gemini_api_key.get_secret_value(),
                convert_system_message_to_human=True
            )
        elif self.provider == "openai":
            return ChatOpenAI(
                model=self.openai_model,
                temperature=self.temperature,
                api_key=self.openai_api_key.get_secret_value(),
                base_url=self.openai_base_url,
                stream_usage=True,
                model_kwargs={
                    "stream_options": {"include_usage": True}
                }  
            )
        raise ValueError(f"Unknown provider: {self.provider}")
