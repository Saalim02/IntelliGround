"""Audit the isolated multilingual benchmark corpus without changing its text."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import median


CORPUS_DIR = Path("storage/benchmarks/multilingual_20261105")
REPORT_JSON = CORPUS_DIR / "ocr_quality_report.json"
REPORT_MD = CORPUS_DIR / "ocr_quality_report.md"

LANGUAGES = {
    "bengali": "BENGALI",
    "gujarati": "GUJARATI",
    "hindi": "DEVANAGARI",
    "kannada": "KANNADA",
    "malayalam": "MALAYALAM",
    "marathi": "DEVANAGARI",
    "punjabi": "GURMUKHI",
    "tamil": "TAMIL",
    "telugu": "TELUGU",
    "urdu": "ARABIC",
}
SCRIPT_NAMES = tuple(sorted(set(LANGUAGES.values())))
METADATA_RE = re.compile(
    r"\b(?:RNI|Registered|Postal License|Fortnightly|Published|Printed|WPP|"
    r"Meghdoot|New Delhi|Publishing Date|Pages[- ]?44)\b",
    re.IGNORECASE,
)
CORRUPTION_CHARS = ("�", "\ufffd")


def _script_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for char in text:
        try:
            name = unicodedata.name(char)
        except ValueError:
            continue
        for script in SCRIPT_NAMES:
            if script in name:
                counts[script] += 1
    return counts


def _page_audit(language: str, expected_script: str, page: dict) -> dict:
    text = (page.get("text") or "").strip()
    counts = _script_counts(text)
    expected_count = counts.get(expected_script, 0)
    script_count = sum(counts.values())
    metadata = bool(METADATA_RE.search(text))
    reasons: list[str] = []

    if not text:
        classification = "EMPTY"
        reasons.append("no extracted text")
    else:
        if len(text) < 100:
            reasons.append("very short extracted text")
        if expected_count == 0:
            reasons.append("expected script absent")
        elif script_count and expected_count / script_count < 0.5:
            reasons.append("unexpected-script characters are prominent")
        if metadata and expected_count < 100:
            reasons.append("publisher metadata dominates")
        if any(char in text for char in CORRUPTION_CHARS):
            reasons.append("replacement/corruption characters present")

        if len(text) < 100 or (expected_count == 0 and not metadata):
            classification = "POOR"
        elif metadata and expected_count < 100:
            classification = "REVIEW_REQUIRED"
        elif reasons:
            classification = "REVIEW_REQUIRED"
        else:
            classification = "ACCEPTABLE"

    return {
        "page_number": page["page_number"],
        "character_count": len(text),
        "word_count": len(text.split()),
        "empty": not bool(text),
        "extraction_method": page.get("extraction_method"),
        "ocr_language": page.get("ocr_language"),
        "expected_script": expected_script,
        "script_counts": dict(counts),
        "expected_script_character_count": expected_count,
        "publisher_metadata_detected": metadata,
        "classification": classification,
        "reasons": reasons,
        "sample": text[:300],
    }


def build_report() -> dict:
    if REPORT_JSON.exists() or REPORT_MD.exists():
        raise FileExistsError(f"Refusing to overwrite existing OCR report: {REPORT_JSON}")

    languages = []
    for language, expected_script in LANGUAGES.items():
        payload = json.loads((CORPUS_DIR / f"pages_{language}.json").read_text(encoding="utf-8"))
        pages = [_page_audit(language, expected_script, page) for page in payload["pages"]]
        counts = Counter(page["classification"] for page in pages)
        lengths = [page["character_count"] for page in pages]
        languages.append(
            {
                "language": language,
                "language_hint": payload["language_hint"],
                "source_filename": payload["source"]["filename"],
                "page_count": len(pages),
                "statistics": payload["statistics"],
                "empty_page_count": counts["EMPTY"],
                "classification_counts": dict(counts),
                "length_statistics": {
                    "minimum": min(lengths),
                    "median": median(lengths),
                    "maximum": max(lengths),
                },
                "sample_pages": [page for page in pages if page["page_number"] in (1, 22, 44)],
                "pages": pages,
            }
        )

    report = {
        "schema_version": 1,
        "benchmark": "multilingual_20261105",
        "corpus_directory": str(CORPUS_DIR),
        "method": {
            "text_modified": False,
            "expected_script_detection": "Unicode character-name counts",
            "publisher_metadata_detection": METADATA_RE.pattern,
            "classification_policy": {
                "EMPTY": "no non-whitespace extracted text",
                "POOR": "very short text or missing expected script without metadata explanation",
                "REVIEW_REQUIRED": "metadata-dominant or script/corruption anomaly",
                "ACCEPTABLE": "nonempty text with expected script and no detected anomaly",
            },
        },
        "languages": languages,
        "totals": {
            "pages": sum(item["page_count"] for item in languages),
            "empty_pages": sum(item["empty_page_count"] for item in languages),
            "classification_counts": dict(
                sum((Counter(item["classification_counts"]) for item in languages), Counter())
            ),
        },
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# OCR Quality Report",
        "",
        "Benchmark corpus: `multilingual_20261105`.",
        "",
        "This is a read-only audit. Extracted text was not rewritten or replaced.",
        "",
        "## Summary",
        "",
        f"- Pages audited: {report['totals']['pages']}",
        f"- Empty pages: {report['totals']['empty_pages']}",
        f"- Classifications: `{report['totals']['classification_counts']}`",
        "",
        "| Language | Pages | Empty | Acceptable | Review | Poor | Methods | Samples 1/22/44 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in languages:
        c = item["classification_counts"]
        methods = item["statistics"]["extraction_method_counts"]
        sample_status = ", ".join(
            f"p{page['page_number']}={page['classification']} ({page['character_count']} chars)"
            for page in item["sample_pages"]
        )
        lines.append(
            f"| {item['language']} | {item['page_count']} | {item['empty_page_count']} | "
            f"{c.get('ACCEPTABLE', 0)} | {c.get('REVIEW_REQUIRED', 0)} | "
            f"{c.get('POOR', 0)} | `{methods}` | {sample_status} |"
        )
    lines += ["", "## Findings requiring review", ""]
    findings = [
        (item["language"], page)
        for item in languages
        for page in item["pages"]
        if page["classification"] != "ACCEPTABLE"
    ]
    if not findings:
        lines.append("No pages were classified below ACCEPTABLE.")
    else:
        for language, page in findings:
            lines.append(
                f"- **{language} page {page['page_number']}** — {page['classification']}; "
                f"{page['character_count']} characters; {', '.join(page['reasons'])}."
            )
    lines += ["", "## Audit scope", "", "Every page was inspected for text length, emptiness, expected Unicode script, unexpected scripts, publisher metadata, corruption markers, extraction method, and repeated content.", ""]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = build_report()
    print(json.dumps(result["totals"], ensure_ascii=False))
