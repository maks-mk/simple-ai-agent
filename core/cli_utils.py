from core.text_utils import (
    TokenTracker,
    abbreviate_path,
    clean_markdown_text,
    format_exception_friendly,
    format_tool_display,
    format_tool_output,
    normalize_markdown_code_blocks,
    parse_thought,
    prepare_markdown_for_render,
    truncate_value,
)


def get_key_bindings():
    """Настройка Alt+Enter для переноса строки."""
    from prompt_toolkit.key_binding import KeyBindings

    kb = KeyBindings()

    @kb.add('enter')
    def _(event):
        buf = event.current_buffer
        # Allow empty Enter through so callers that treat "" as a default
        # (e.g. approval prompt with default-yes) work correctly.
        # Only suppress bare Enter in the *main* chat loop — that's handled
        # by the `if not user_input: continue` guard in agent_cli.py.
        buf.validate_and_handle()

    @kb.add('escape', 'enter')
    def _(event):
        event.current_buffer.insert_text("\n")

    return kb


