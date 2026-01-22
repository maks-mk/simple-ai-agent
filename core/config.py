import sys
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- LANGCHAIN PROVIDERS ---
# Импортируем провайдеров здесь, чтобы не засорять основной файл
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

class AgentConfig(BaseSettings):
    # Универсальный способ найти папку с конфигами
    if getattr(sys, 'frozen', False):
        _base_dir: Path = Path(sys.executable).parent
    else:
        _base_dir: Path = Path(__file__).resolve().parent.parent
        
    # Указываем полные пути к конфигам, опираясь на _base_dir
    model_config = SettingsConfigDict(
        env_file= _base_dir / '.env',
        env_file_encoding='utf-8', 
        extra='ignore'
    )
    
    # Paths
    prompt_path: Path = Field(default=_base_dir / "prompt.txt", alias="PROMPT_PATH")
    mcp_config_path: Path = _base_dir / "mcp.json"
    memory_db_path: str = str(_base_dir / "memory_db")
    
    provider: Literal["gemini", "openai"] = "gemini"
    
    # API Keys & Models
    gemini_api_key: Optional[SecretStr] = None
    gemini_model: str = "gemini-1.5-flash"
    
    openai_api_key: Optional[SecretStr] = None
    openai_model: str = "gpt-4o"
    openai_base_url: Optional[str] = None

    temperature: float = 0.2
    
    # Logic Settings
    enable_deep_search: bool = Field(default=False, alias="DEEP_SEARCH")
    model_supports_tools: bool = Field(default=True, alias="MODEL_SUPPORTS_TOOLS")
    use_long_term_memory: bool = Field(default=False, alias="LONG_TERM_MEMORY")
    max_loops: int = Field(default=15, description="Limit steps per request")
    
    # Summarization Settings
    summary_threshold: int = Field(default=20, alias="SESSION_SIZE")
    summary_keep_last: int = Field(default=4, alias="SUMMARY_KEEP_LAST")

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
            
        # Эвристика для OpenRouter и других провайдеров
        if self.provider == "openai":
            model_name = self.openai_model.lower()
            no_tool_prefixes = (
                "tngtech/", "huggingface/", "grey-wing/", "sao10k/" 
            )
            if model_name.startswith(no_tool_prefixes):
                return False
        return True

    def get_llm(self) -> BaseChatModel:
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