
from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache


EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

DOCUMENT_PROMPT = "passage: "
QUERY_PROMPT = "query: "

NORMALIZE_EMBEDDINGS = True


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Returns multilingual E5 embeddings.

    Documents use the passage prefix.
    Queries use the query prefix.
    Both use normalized vectors.
    """

    return HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        encode_kwargs={
            "normalize_embeddings": NORMALIZE_EMBEDDINGS,
            "prompt": DOCUMENT_PROMPT,
        },
        query_encode_kwargs={
            "normalize_embeddings": NORMALIZE_EMBEDDINGS,
            "prompt": QUERY_PROMPT,
        },
    )
