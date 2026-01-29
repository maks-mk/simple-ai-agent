@echo off
chcp 65001 > nul

pyinstaller --name "ai_аgent" --onefile --clean ^
 --collect-all tiktoken ^
 --collect-all langchain ^
 --collect-all rich ^
 --hidden-import=tiktoken_ext ^
 --hidden-import=tiktoken_ext.openai_public ^
 --hidden-import=pydantic.deprecated.decorator ^
 agent_cli.py

pause