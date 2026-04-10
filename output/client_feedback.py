"""
Client feedback output formatter.
Generates structured markdown grouped by category (BUGs, WISHes, POSITIVEs, QUESTIONs).
"""
from __future__ import annotations

from typing import Any

from core.correlator import CorrelationResult, FeedbackItem


def format_client_feedback(
    correlation: CorrelationResult,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Format CorrelationResult into structured client feedback markdown."""
    metadata = metadata or {}
    lines: list[str] = []

    # Header
    source = metadata.get("source", "Unknown source")
    lines.append("# Клиентский фидбек — автоматический анализ")
    lines.append("")
    lines.append(f"**Источник:** `{source}`")
    if "duration" in metadata:
        dur = metadata["duration"]
        m = int(dur // 60)
        s = int(dur % 60)
        lines.append(f"**Длительность:** {m} мин {s} сек")
    lines.append(f"**Режим:** автоматическая корреляция аудио + экран")
    lines.append("")

    # Group items by category
    bugs = [i for i in correlation.feedback_items if i.category == "BUG"]
    wishes = [i for i in correlation.feedback_items if i.category == "WISH"]
    questions = [i for i in correlation.feedback_items if i.category == "QUESTION"]

    # Bugs
    if bugs:
        lines.append("---")
        lines.append("")
        lines.append("## P0 — Критические баги")
        lines.append("")
        for item in bugs:
            lines.extend(_format_item(item))

    # Wishes
    if wishes:
        lines.append("---")
        lines.append("")
        lines.append("## P1 — Запросы на фичи")
        lines.append("")
        for item in wishes:
            lines.extend(_format_item(item))

    # Questions
    if questions:
        lines.append("---")
        lines.append("")
        lines.append("## P2 — Под вопросом")
        lines.append("")
        for item in questions:
            lines.extend(_format_item(item))

    # Positives
    if correlation.positives:
        lines.append("---")
        lines.append("")
        lines.append("## Что нравится — НЕ ТРОГАТЬ")
        lines.append("")
        lines.append("| Элемент | Кадр | Цитата |")
        lines.append("|---|---|---|")
        for pos in correlation.positives:
            element = pos.get("element", "")
            frame = pos.get("frame", "")
            quote = pos.get("quote", "")
            time = pos.get("time", "")
            time_str = f" ({time})" if time else ""
            lines.append(f"| {element} | `{frame}`{time_str} | {quote} |")
        lines.append("")

    # Summary table
    all_items = correlation.feedback_items
    if all_items:
        lines.append("---")
        lines.append("")
        lines.append("## Сводная приоритезация")
        lines.append("")
        lines.append("| # | Задача | Приоритет | Категория |")
        lines.append("|---|---|---|---|")
        for item in all_items:
            lines.append(f"| {item.id} | {item.title} | {item.priority} | {item.category} |")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Сгенерировано автоматически VVAM (Video Voice AI Manager) — корреляция аудио-речи с экраном*")
    lines.append("")

    return "\n".join(lines)


def _format_item(item: FeedbackItem) -> list[str]:
    """Format a single feedback item."""
    lines: list[str] = []

    lines.append(f"### {item.id}. {item.title}")
    lines.append("")
    lines.append(f"**Описание:** {item.description}")
    lines.append("")

    # Numeric conflicts
    if item.numeric_conflicts:
        lines.append("**Конфликт значений (речь vs экран):**")
        for nc in item.numeric_conflicts:
            speech = nc.get("speech_value", "?")
            screen = nc.get("screen_value", "?")
            entity = nc.get("entity", "?")
            lines.append(f"- {entity}: речь говорит **{speech}**, экран показывает **{screen}**")
        lines.append("")

    # Quotes
    if item.quotes:
        lines.append("**Цитаты:**")
        for q in item.quotes:
            time = q.get("time", "??:??")
            text = q.get("text", "")
            lines.append(f"- {time} — «{text}»")
        lines.append("")

    # Frame refs
    if item.frame_refs:
        lines.append("**Визуальные доказательства:**")
        for fr in item.frame_refs:
            frame = fr.get("frame", "")
            time = fr.get("time", "")
            desc = fr.get("description", "")
            time_str = f" ({time})" if time else ""
            lines.append(f"- `{frame}`{time_str} — {desc}")
        lines.append("")

    # Action needed
    if item.action_needed:
        lines.append(f"**Что нужно:** {item.action_needed}")
        lines.append("")

    return lines
