"""
Formatter factory — returns the appropriate output formatter by name.
"""
from __future__ import annotations

from typing import Callable

from output.markdown import format_markdown
from output.json_out import format_json
from output.srt import format_srt

_FORMATTERS: dict[str, Callable] = {
    "markdown": format_markdown,
    "json": format_json,
    "srt": format_srt,
}

# client-feedback is special — handled separately in cmd_video
# because it takes CorrelationResult, not transcript+frames


def get_formatter(name: str) -> Callable:
    """
    Get an output formatter by name.

    Supported names: "markdown", "json", "srt".
    Raises ValueError for unknown format names.
    """
    fmt = _FORMATTERS.get(name)
    if fmt is None:
        supported = ", ".join(sorted(_FORMATTERS))
        raise ValueError(f"Unknown format {name!r}. Supported: {supported}")
    return fmt
