
"""Multilingual, persistence-friendly BM25 retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    # Production implementation.
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None


SPARSE_INDEX_FILENAME = "sparse_index.json"
SPARSE_INDEX_SCHEMA_VERSION = 1


def tokenize(text: str | None) -> list[str]:
    """
    Normalize Unicode text and tokenize letters, numbers and combining marks.

    This is a script-preserving baseline tokenizer. It does not perform
    language-specific stemming, stop-word removal or morphological analysis.
    """
    if not text:
        return []

    normalized = unicodedata.normalize("NFC", str(text))

    tokens: list[str] = []
    current: list[str] = []

    for char in normalized:
        category = unicodedata.category(char)

        if category[0] in {"L", "N"} or category.startswith("M"):
            current.append(char.casefold() if char.isascii() else char)

        elif current:
            tokens.append("".join(current))
            current = []

    if current:
        tokens.append("".join(current))

    return tokens


def document_identity(
    metadata: dict[str, Any],
    text: str = "",
) -> str:
    """
    Return a stable identity shared by dense, sparse and fused results.

    Preferred identity:
        doc_id + chunk_id

    Fallback:
        Metadata fields + text hash
    """
    doc_id = metadata.get("doc_id")
    chunk_id = metadata.get("chunk_id")

    if doc_id is not None and chunk_id is not None:
        return f"{doc_id}:{chunk_id}"

    fallback_fields = (
        "file_name",
        "page_number",
        "page_chunk_id",
        "start_char",
        "end_char",
    )

    metadata_identity = ":".join(
        str(metadata.get(field, ""))
        for field in fallback_fields
    )

    text_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return f"{metadata_identity}:{text_hash}"


@dataclass
class SparseRecord:
    """One document chunk stored in the sparse index."""

    text: str
    metadata: dict[str, Any]
    tokens: list[str]


class _FallbackBM25:
    """
    Lightweight BM25 implementation used when rank-bm25 is unavailable.

    Parameters follow the standard BM25 configuration:
        k1 = 1.5
        b = 0.75
    """

    def __init__(
        self,
        corpus: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b

        self.document_count = len(corpus)

        self.avgdl = (
            sum(len(tokens) for tokens in corpus) / self.document_count
            if self.document_count
            else 0.0
        )

        self.df: dict[str, int] = {}

        self.term_frequencies: list[dict[str, int]] = []

        for tokens in corpus:
            frequencies: dict[str, int] = {}

            for term in tokens:
                frequencies[term] = frequencies.get(term, 0) + 1

            self.term_frequencies.append(frequencies)

            for term in frequencies:
                self.df[term] = self.df.get(term, 0) + 1

    def get_scores(
        self,
        query_tokens: list[str],
    ) -> list[float]:
        """Calculate BM25 scores for every document."""

        scores = [0.0] * self.document_count

        if not self.document_count:
            return scores

        for index, frequencies in enumerate(self.term_frequencies):
            document_length = len(self.corpus[index])

            length_normalization = (
                self.k1
                * (
                    1 - self.b
                    + self.b * document_length / (self.avgdl or 1)
                )
            )

            for term in query_tokens:
                term_frequency = frequencies.get(term, 0)

                if term_frequency == 0:
                    continue

                document_frequency = self.df.get(term, 0)

                idf = math.log(
                    1
                    + (
                        self.document_count
                        - document_frequency
                        + 0.5
                    )
                    / (
                        document_frequency + 0.5
                    )
                )

                numerator = term_frequency * (self.k1 + 1)

                denominator = (
                    term_frequency
                    + length_normalization
                )

                scores[index] += (
                    idf * numerator / denominator
                )

        return scores


class BM25Index:
    """In-memory BM25 index with JSON persistence support."""

    def __init__(
        self,
        records: list[SparseRecord],
    ) -> None:
        self.records = records

        corpus = [
            record.tokens
            for record in records
        ]

        if BM25Okapi is not None and corpus:
            self._index = BM25Okapi(corpus)
        else:
            self._index = _FallbackBM25(corpus)

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Any],
    ) -> "BM25Index":
        """
        Build a sparse index from LangChain Documents or dictionaries.

        Supported inputs:

        LangChain Document:
            document.page_content
            document.metadata

        Dictionary:
            document["page_content"]
            document["metadata"]
        """
        records: list[SparseRecord] = []

        for document in documents:

            if isinstance(document, dict):
                text = str(
                    document.get("page_content") or ""
                )

                metadata = dict(
                    document.get("metadata") or {}
                )

            else:
                text = str(
                    getattr(document, "page_content", "") or ""
                )

                metadata = dict(
                    getattr(document, "metadata", None) or {}
                )

            records.append(
                SparseRecord(
                    text=text,
                    metadata=metadata,
                    tokens=tokenize(text),
                )
            )

        return cls(records)

    def search(
        self,
        query: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return the highest-scoring matching document chunks."""

        if limit < 0:
            raise ValueError(
                "Search limit cannot be negative."
            )

        query_tokens = tokenize(query)

        if not self.records or not query_tokens or limit == 0:
            return []

        scores = self._index.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (
                -float(scores[index]),
                index,
            ),
        )

        results: list[dict[str, Any]] = []

        for index in ranked_indices:

            score = float(scores[index])

            # Do not return documents without a lexical match.
            if not math.isfinite(score) or score <= 0:
                continue

            record = self.records[index]

            results.append(
                {
                    "identity": document_identity(
                        record.metadata,
                        record.text,
                    ),
                    "text": record.text,
                    "metadata": dict(record.metadata),
                    "score": score,
                    "rank": len(results) + 1,
                }
            )

            if len(results) >= limit:
                break

        return results


def save_sparse_index(
    index: BM25Index,
    directory: Path,
) -> None:
    """Persist sparse index records as UTF-8 JSON."""

    directory = Path(directory)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": SPARSE_INDEX_SCHEMA_VERSION,
        "records": [
            {
                "text": record.text,
                "metadata": record.metadata,
                "tokens": record.tokens,
            }
            for record in index.records
        ],
    }

    path = directory / SPARSE_INDEX_FILENAME

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
        )


def load_sparse_index(
    directory: Path,
) -> BM25Index | None:
    """
    Load and validate a persisted sparse index.

    Returns None when:
        - File is missing.
        - JSON is corrupted.
        - Schema version is incompatible.
        - Record structure is invalid.
        - Stored tokens do not match current tokenization.
    """

    path = Path(directory) / SPARSE_INDEX_FILENAME

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            return None

        if payload.get("schema_version") != SPARSE_INDEX_SCHEMA_VERSION:
            return None

        raw_records = payload.get("records")

        if not isinstance(raw_records, list):
            return None

        records: list[SparseRecord] = []

        for item in raw_records:

            if not isinstance(item, dict):
                return None

            text = item.get("text")
            metadata = item.get("metadata")
            tokens = item.get("tokens")

            if not isinstance(text, str):
                return None

            if not isinstance(metadata, dict):
                return None

            if not isinstance(tokens, list):
                return None

            if not all(isinstance(token, str) for token in tokens):
                return None

            # Reject indexes generated with a different tokenizer.
            if tokenize(text) != tokens:
                return None

            records.append(
                SparseRecord(
                    text=text,
                    metadata=dict(metadata),
                    tokens=list(tokens),
                )
            )

        return BM25Index(records)

    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
    ):
        return None
