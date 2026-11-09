
from collections import Counter

from app.services.language_registry import LANGUAGE_CODES, LANGUAGE_REGISTRY


VOICE_RESPONSE_LANGUAGES = frozenset(LANGUAGE_CODES)


LANGUAGE_RANGES = {
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "te": (0x0C00, 0x0C7F),  # Telugu
    "hi": (0x0900, 0x097F),  # Devanagari
    "ml": (0x0D00, 0x0D7F),  # Malayalam
    "bn": (0x0980, 0x09FF),  # Bengali
    "gu": (0x0A80, 0x0AFF),  # Gujarati
    "pa": (0x0A00, 0x0A7F),  # Gurmukhi
    "ur": (0x0600, 0x06FF),  # Arabic-derived Urdu script
    "or": (0x0B00, 0x0B7F),  # Odia
}


def detect_text_language(text: str) -> dict:
    """
    Detect dominant writing system using Unicode ranges.

    This identifies scripts, not semantic language.
    Latin text is labelled 'en' as a practical default,
    not as proof that the text is English.
    """

    counts = Counter()
    total_letters = 0

    for char in text:
        if not char.isalpha():
            continue

        total_letters += 1
        codepoint = ord(char)

        matched = False

        for language, (start, end) in LANGUAGE_RANGES.items():
            if start <= codepoint <= end:
                counts[language] += 1
                matched = True
                break

        if not matched and (
            0x0041 <= codepoint <= 0x024F
        ):
            counts["en"] += 1

    if total_letters == 0:
        return {
            "language": "und",
            "confidence": 0.0,
            "script_counts": {},
        }

    ranked = counts.most_common()

    if not ranked:
        language = "und"
        confidence = 0.0
    else:
        dominant_language, dominant_count = ranked[0]
        confidence = dominant_count / total_letters

        if len(ranked) > 1:
            second_count = ranked[1][1]

            # Avoid assigning one language when two scripts
            # have substantial representation.
            if second_count / total_letters >= 0.20:
                language = "mixed"
            else:
                language = dominant_language
        else:
            language = dominant_language

        if confidence < 0.60:
            language = "mixed"

    return {
        "language": language,
        "confidence": round(confidence, 3),
        "script_counts": dict(counts),
    }


def _normalize_provider_language(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("_", "-")
    if normalized in LANGUAGE_REGISTRY:
        return normalized
    base = normalized.split("-", 1)[0]
    return base if base in LANGUAGE_REGISTRY else None


def detect_voice_response_language(
    text: str,
    previous_language: str | None = None,
    provider_language: str | None = None,
) -> str:
    """Choose the authoritative spoken response language for a voice turn.

    Dominant script drives switching. Short isolated Latin technical terms,
    numbers, and low-confidence mixed transcripts retain the previous voice
    language instead of causing an accidental switch.
    """
    previous = (previous_language or "en").split("-", 1)[0].lower()
    if previous not in VOICE_RESPONSE_LANGUAGES:
        previous = "en"
    provider = _normalize_provider_language(provider_language)
    if provider in VOICE_RESPONSE_LANGUAGES:
        return provider
    info = detect_text_language(text)
    detected = info["language"]
    counts = info.get("script_counts", {})
    total = sum(counts.values())
    if not total:
        return previous
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    dominant, dominant_count = ranked[0]
    dominant_ratio = dominant_count / total
    if detected == previous:
        return previous
    if dominant not in VOICE_RESPONSE_LANGUAGES:
        return previous
    # A short isolated Latin technical term such as "API" does not constitute
    # a genuine switch.  A short utterance in a distinct Indic script can be
    # authoritative when Prisma did not provide metadata.
    if dominant == "en" and sum(char.isalpha() for char in text) < 12:
        return previous
    if dominant_ratio < 0.60:
        return previous
    return dominant
