@echo off
chcp 65001 > nul
echo ========================================
echo Optimized Build for AI Agent
echo ========================================

:: Check if UPX is available
where upx >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] UPX found - will compress executable
    set USE_UPX=
) else (
    echo [!] UPX not found - skipping compression
    echo [!] Install UPX from https://upx.github.io/ for smaller binaries
    set USE_UPX=--noupx
)

:: Build with optimizations
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
 --exclude-module=IPython ^
 --exclude-module=jupyter ^
 --strip ^
 --optimize=2 ^
 %USE_UPX% ^
 --icon=icon.ico ^
 agent_cli.py

echo.
echo ========================================
echo Build complete!
echo ========================================
echo Output: dist\ai_agent.exe
echo.
if not "%USE_UPX%"=="--noupx" (
    echo [✓] Binary compressed with UPX
)

pause