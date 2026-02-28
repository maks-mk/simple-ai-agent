@echo off
chcp 65001 > nul

pyinstaller --name "ai_agent" --onefile --clean ^
 --collect-all tiktoken ^
 --collect-all langchain ^
 --collect-all langchain_openai ^
 --collect-all langchain_google_genai ^
 --collect-all rich ^
 --collect-all prompt_toolkit ^
 --collect-all pygments ^
 --hidden-import=tiktoken_ext ^
 --hidden-import=tiktoken_ext.openai_public ^
 --hidden-import=langchain_core.tools ^
 --hidden-import=langchain_core.messages ^
 --hidden-import=langgraph.checkpoint.memory ^
 --exclude-module=tkinter ^
 --exclude-module=matplotlib ^
 --exclude-module=pandas ^
 --exclude-module=numpy ^
 --strip ^
 --optimize=2 ^
 --noupx ^
 --icon=icon.ico ^
 agent_cli.py

pause