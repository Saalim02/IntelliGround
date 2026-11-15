"""Shared response-length policy for grounded answer generation."""

from typing import Literal

ResponseLength = Literal["short", "medium", "large"]
DEFAULT_RESPONSE_LENGTH: ResponseLength = "medium"

RESPONSE_LENGTH_INSTRUCTIONS: dict[str, str] = {
    "short": (
        "Aim for 2–3 readable lines. Answer the exact question directly, include "
        "essential supporting evidence and citations, avoid introductions and repetition, "
        "and do not omit a necessary qualification."
    ),
    "medium": (
        "Aim for 4–6 readable lines. Provide the answer, important supporting details, "
        "a concise explanation, and relevant citations with page references."
    ),
    "large": (
        "Produce more than 15 readable lines when the retrieved document contains enough "
        "supporting material. Explain the topic thoroughly with headings or bullets where "
        "useful, important facts, examples and context, multiple supporting citations and "
        "page references, and limitations or missing evidence. Avoid repetition and "
        "unsupported elaboration; never invent evidence to make the answer longer."
    ),
}

# These caps constrain generation itself; no answer is truncated after generation.
RESPONSE_LENGTH_MAX_TOKENS: dict[str, int] = {"short": 256, "medium": 512, "large": 1536}

DOCUMENT_INTELLIGENCE_TARGETS: dict[str, dict[str, int]] = {
    "short": {"topics": 5, "questions": 5, "questions_max": 5},
    "medium": {"topics": 10, "questions": 10, "questions_max": 10},
    "large": {"topics": 15, "questions": 15, "questions_max": 20},
}


def document_generation_plan(chunks: list[dict], value: str | None) -> dict[str, int | str]:
    """Compute grounded quantity targets from evidence, not page count alone."""
    length = value if value in DOCUMENT_INTELLIGENCE_TARGETS else DEFAULT_RESPONSE_LENGTH
    targets = DOCUMENT_INTELLIGENCE_TARGETS[length]
    meaningful = [
        chunk for chunk in chunks
        if sum(char.isalpha() for char in str(chunk.get("text", ""))) >= 20
    ]
    section_keys = {
        str(chunk.get("section") or chunk.get("section_title") or chunk.get("topic")
            or f"page:{chunk.get('page_number', 'unknown')}")
        for chunk in meaningful
    }
    section_count = len(section_keys) if meaningful else 0
    capacity = max(0, min(targets["questions_max"], len(meaningful) // 2 + section_count))
    return {
        "response_length": length,
        "meaningful_chunks": len(meaningful),
        "section_count": section_count,
        "total_pages": len({chunk.get("page_number") for chunk in chunks if chunk.get("page_number") is not None}),
        "target_topics": min(targets["topics"], capacity),
        "target_questions": min(targets["questions"], capacity),
        "max_questions": min(targets["questions_max"], capacity),
    }


def response_length_instruction(value: str | None) -> str:
    return RESPONSE_LENGTH_INSTRUCTIONS.get(value or DEFAULT_RESPONSE_LENGTH, RESPONSE_LENGTH_INSTRUCTIONS[DEFAULT_RESPONSE_LENGTH])


def response_length_max_tokens(value: str | None) -> int:
    return RESPONSE_LENGTH_MAX_TOKENS.get(value or DEFAULT_RESPONSE_LENGTH, RESPONSE_LENGTH_MAX_TOKENS[DEFAULT_RESPONSE_LENGTH])
