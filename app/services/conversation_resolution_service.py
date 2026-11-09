"""Deterministic, user-scoped follow-up reference resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedFollowUp:
    query: str
    page_numbers: tuple[int, ...] = ()
    ambiguous: bool = False


def _context_pages(context: str) -> tuple[int, ...]:
    pages: list[int] = []
    for match in re.finditer(r"Assistant evidence pages:\s*([0-9, ]+)", context or "", re.IGNORECASE):
        for value in re.findall(r"\d+", match.group(1)):
            page = int(value)
            if page not in pages:
                pages.append(page)
    return tuple(pages)


def resolve_follow_up(question: str, context: str | None) -> ResolvedFollowUp:
    original = (question or "").strip()
    context_text = (context or "").strip()
    if not context_text:
        return ResolvedFollowUp(original, ambiguous=True)

    pages = _context_pages(context_text)
    normalized = re.sub(r"\s+", " ", original.lower())
    if pages and re.search(r"\b(next|अगला|ಮುಂದಿನ|அடுத்த|తదుపరి)\s+page\b", normalized):
        return ResolvedFollowUp(original, (max(pages) + 1,))
    return ResolvedFollowUp(
        f"Conversation context:\n{context_text}\n\nCurrent user question:\n{original}"
    )
