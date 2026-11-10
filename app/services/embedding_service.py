
import os
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_PROVIDER = "huggingface"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_MODEL_VERSION = os.getenv("EMBEDDING_MODEL_VERSION", "main").strip()
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))

DOCUMENT_PROMPT = "passage: "
QUERY_PROMPT = "query: "

NORMALIZE_EMBEDDINGS = True


def _embedding_identity() -> tuple:
    return (
        EMBEDDING_PROVIDER,
        EMBEDDING_MODEL_NAME,
        EMBEDDING_MODEL_VERSION,
        EMBEDDING_DIMENSIONS,
        DOCUMENT_PROMPT,
        QUERY_PROMPT,
        NORMALIZE_EMBEDDINGS,
    )


@lru_cache(maxsize=8)
def _get_embedding_model_cached(
    provider: str,
    model_name: str,
    model_version: str,
    dimensions: int,
    document_prompt: str,
    query_prompt: str,
    normalize_embeddings: bool,
):
    """
    Returns multilingual E5 embeddings.

    Documents use the passage prefix.
    Queries use the query prefix.
    Both use normalized vectors.
    """

    if provider != "huggingface":
        raise ValueError(f"Unsupported embedding provider: {provider}")
    return HuggingFaceEmbeddings(
        model=model_name,
        model_kwargs={"revision": model_version},
        encode_kwargs={
            "normalize_embeddings": normalize_embeddings,
            "prompt": document_prompt,
        },
        query_encode_kwargs={
            "normalize_embeddings": normalize_embeddings,
            "prompt": query_prompt,
        },
    )


def get_embedding_model():
    return _get_embedding_model_cached(*_embedding_identity())


@lru_cache(maxsize=2048)
def _embed_query_cached(query: str, *identity) -> tuple[float, ...]:
    """Reuse embeddings only within one complete model configuration."""
    vector = tuple(_get_embedding_model_cached(*identity).embed_query(query))
    expected_dimensions = identity[3]
    if len(vector) != expected_dimensions:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dimensions}, got {len(vector)}"
        )
    return vector


def embed_query_cached(query: str) -> tuple[float, ...]:
    return _embed_query_cached(query, *_embedding_identity())
