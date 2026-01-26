import sys
from pathlib import Path

# Определение корневой директории проекта
# Это позволяет корректно работать как в режиме скрипта, так и в скомпилированном EXE
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    # core/constants.py -> core/ -> root/
    BASE_DIR = Path(__file__).resolve().parent.parent
