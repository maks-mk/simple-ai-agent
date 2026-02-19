import os
from pathlib import Path
from typing import Iterable, List
from difflib import SequenceMatcher
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

class FuzzyPathCompleter(Completer):
    """
    A completer that fuzzy matches file paths in the current directory and subdirectories.
    """
    def __init__(self, root_dir: str = ".", max_depth: int = 4, min_score: float = 0.4):
        self.root_dir = Path(root_dir).resolve()
        self.max_depth = max_depth
        self.min_score = min_score
        self._cache_files: List[str] = []
        self._cache_valid = False

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        text = document.text_before_cursor
        
        # Simple heuristic: only trigger if typing something that looks like an argument (space before)
        # or if it's the first word (command or script)
        word = document.get_word_before_cursor(WORD=True)
        if not word:
            return

        # Refresh cache if needed (lazy load)
        if not self._cache_valid:
            self._refresh_file_cache()

        candidates = []
        # Normalize input for matching (handle Windows backslashes and ./ prefix)
        word_lower = word.lower().replace("\\", "/")
        if word_lower.startswith("./"):
            word_lower = word_lower[2:]
        
        for file_path in self._cache_files:
            fp_lower = file_path.lower()
            
            # 1. Exact Prefix Match (High Priority)
            if fp_lower.startswith(word_lower):
                # Score 1.0 + length bonus (shorter is better)
                score = 1.0 + (1.0 / (len(file_path) + 1))
                candidates.append((score, file_path))
                continue
                
            # 2. Subsequence / Fuzzy Match
            # Check if all chars in word appear in file_path in order (subsequence check)
            # This is faster than difflib for filtering
            if self._is_subsequence(word_lower, fp_lower):
                # Calculate detailed score
                ratio = SequenceMatcher(None, word_lower, fp_lower).ratio()
                if ratio >= self.min_score:
                    candidates.append((ratio, file_path))

        # Sort by score desc
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Yield top 15
        for score, file_path in candidates[:15]:
            # Highlight matched part? prompt_toolkit handles display mostly
            # We use negative start_position to replace the word typed so far
            yield Completion(
                file_path, 
                start_position=-len(word), 
                display=file_path, 
                display_meta=f"File ({score:.2f})"
            )

    def _is_subsequence(self, query: str, target: str) -> bool:
        """Check if query is a subsequence of target."""
        it = iter(target)
        return all(char in it for char in query)

    def _refresh_file_cache(self):
        self._cache_files = []
        try:
            # Simple walk with depth limit
            # Use os.walk for efficiency
            start_dir = str(self.root_dir)
            root_depth = len(self.root_dir.parts)
            
            for root, dirs, files in os.walk(start_dir):
                # Check depth
                current_parts = Path(root).parts
                if len(current_parts) - root_depth >= self.max_depth:
                    dirs[:] = [] 
                    continue
                
                # Filter hidden dirs
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('venv', '__pycache__', 'node_modules', '.git')]
                
                # Calculate relative path prefix
                rel_dir = Path(root).relative_to(self.root_dir)
                prefix = "" if str(rel_dir) == "." else str(rel_dir).replace("\\", "/") + "/"
                
                for f in files:
                    if f.startswith('.'): continue
                    self._cache_files.append(prefix + f)
                    
        except Exception:
            pass
        self._cache_valid = True
