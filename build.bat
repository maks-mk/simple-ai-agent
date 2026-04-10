@echo off
setlocal
chcp 65001 > nul

set "SCRIPT_DIR=%~dp0"
set "VENV_PYTHON=%SCRIPT_DIR%venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo [ERROR] venv Python not found: "%VENV_PYTHON%"
    pause
    exit /b 1
)

call "%VENV_PYTHON%" -m PyInstaller ^
--name ai-agent ^
--onefile ^
--clean ^
--collect-all tiktoken ^
--collect-all langgraph ^
--collect-all langchain ^
--collect-all langchain_openai ^
--collect-all langchain_google_genai ^
--hidden-import=rich ^
--collect-submodules tools ^
--hidden-import=tiktoken_ext ^
--hidden-import=tiktoken_ext.openai_public ^
--icon=icon.ico ^
agent_cli.py

set "BUILD_EXIT=%ERRORLEVEL%"

pause
exit /b %BUILD_EXIT%