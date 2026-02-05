@echo off
chcp 65001 > nul

pyinstaller --name "ai_agent" --onefile --clean ^
 --collect-all tiktoken ^
 --collect-all langchain ^
 --collect-all langchain_community ^
 --collect-all langchain_openai ^
 --collect-all langchain_google_genai ^
 --collect-all chromadb ^
 --collect-all sentence_transformers ^
 --collect-all rich ^
 --collect-all dotenv ^
 --hidden-import=tiktoken_ext ^
 --hidden-import=tiktoken_ext.openai_public ^
 --hidden-import=pydantic.deprecated.decorator ^
 --hidden-import=langchain_community.vectorstores ^
 --hidden-import=langchain_community.embeddings ^
 --icon=icon.ico ^
 agent_cli.py

pause