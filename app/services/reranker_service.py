"""Lazy multilingual cross-encoder reranking with explicit status."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, *, enabled: bool | None = None, model_name: str | None = None,
                 max_length: int | None = None, model: Any = None) -> None:
        self.enabled = (os.getenv("RERANKER_ENABLED", "false").lower() == "true") if enabled is None else enabled
        self.model_name = model_name or os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        self.max_length = max_length or int(os.getenv("RERANKER_MAX_LENGTH", "512"))
        self.model = model
        self.status = "disabled" if not self.enabled else "not_loaded"

    def _load(self) -> Any:
        if self.model is not None:
            return self.model
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(self.model_name, max_length=self.max_length, device="cpu")
        return self.model

    def rerank(self, query: str, candidates: list[dict[str, Any]], limit: int | None = None) -> tuple[list[dict[str, Any]], str]:
        if not self.enabled:
            return candidates[:limit] if limit else candidates, "disabled"
        if not candidates:
            return [], "enabled_empty"
        try:
            model = self._load()
            scores = model.predict([(query, item.get("text", "")) for item in candidates], batch_size=8, show_progress_bar=False)
            ranked = []
            for item, score in zip(candidates, scores):
                enriched = dict(item)
                enriched["reranker_score"] = float(score)
                ranked.append(enriched)
            ranked.sort(key=lambda item: (-item["reranker_score"], item["identity"]))
            self.status = "enabled_success"
            return ranked[:limit] if limit else ranked, self.status
        except Exception:
            self.status = "enabled_failed"
            logger.exception("Cross-encoder reranker failed; returning explicit unreranked status")
            return candidates[:limit] if limit else candidates, self.status


_default_reranker = CrossEncoderReranker()


def get_reranker() -> CrossEncoderReranker:
    return _default_reranker
