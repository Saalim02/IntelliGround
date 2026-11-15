
import json
import fcntl
import logging
import shutil
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.services.embedding_service import (
    get_embedding_model,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_VERSION,
    EMBEDDING_DIMENSIONS,
    DOCUMENT_PROMPT,
    QUERY_PROMPT,
    NORMALIZE_EMBEDDINGS,
)

from app.core.config import VECTORSTORE_DIR
from app.services.user_storage_service import get_user_document_vectorstore_dir, get_user_vectorstore_dir
from app.services.bm25_service import (
    BM25Index,
    SPARSE_INDEX_FILENAME,
    load_sparse_index,
    save_sparse_index,
)
from app.services.document_registry_service import get_active_document


logger = logging.getLogger(__name__)


class _BatchedEmbeddings:
    """Keep document embedding calls bounded without changing FAISS metadata."""

    def __init__(self, embeddings, batch_size=EMBEDDING_BATCH_SIZE, progress_callback=None):
        self._embeddings = embeddings
        self._batch_size = max(1, batch_size)
        self._progress_callback = progress_callback

    def embed_documents(self, texts):
        total = len(texts)
        if not total:
            return []
        batch_count = (total + self._batch_size - 1) // self._batch_size
        vectors = []
        for batch_number, start in enumerate(range(0, total, self._batch_size), start=1):
            vectors.extend(self._embeddings.embed_documents(texts[start:start + self._batch_size]))
            if self._progress_callback:
                self._progress_callback(batch_number, batch_count)
        return vectors

    def embed_query(self, text):
        return self._embeddings.embed_query(text)

    def __getattr__(self, name):
        return getattr(self._embeddings, name)


def _embedding_adapter(embeddings, progress_callback=None):
    # Keep compatibility with small test doubles and alternate LangChain
    # implementations that only expose FAISS-compatible constructors.
    if not hasattr(embeddings, "embed_documents"):
        return embeddings
    return _BatchedEmbeddings(embeddings, progress_callback=progress_callback)


# --------------------------------------------------
# EMBEDDING MODEL MANIFEST
# --------------------------------------------------

EMBEDDING_MANIFEST_FILENAME = "embedding_model.json"


def _get_embedding_manifest() -> dict:
    """
    Returns the embedding configuration used to create
    the FAISS vectorstore.
    """

    return {
        "schema_version": 1,
        "provider": EMBEDDING_PROVIDER,
        "model_name": EMBEDDING_MODEL_NAME,
        "model_version": EMBEDDING_MODEL_VERSION,
        "dimensions": EMBEDDING_DIMENSIONS,
        "document_prompt": DOCUMENT_PROMPT,
        "query_prompt": QUERY_PROMPT,
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
    }


def _write_embedding_manifest(vectorstore_dir: Path) -> None:
    """
    Writes embedding configuration into the vectorstore directory.
    """

    manifest_path = (
        vectorstore_dir / EMBEDDING_MANIFEST_FILENAME
    )

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            _get_embedding_manifest(),
            file,
            indent=2,
        )


def _has_compatible_embedding_manifest(
    vectorstore_dir: Path,
) -> bool:
    """
    Checks whether the saved vectorstore was created
    using the current embedding configuration.

    Missing, corrupted or mismatched manifests are rejected.
    """

    manifest_path = (
        vectorstore_dir / EMBEDDING_MANIFEST_FILENAME
    )

    if not manifest_path.is_file():
        return False

    try:
        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            saved_manifest = json.load(file)

        return saved_manifest == _get_embedding_manifest()

    except (OSError, json.JSONDecodeError):
        return False


# --------------------------------------------------
# CHUNK CONVERSION
# --------------------------------------------------

def convert_chunks_to_documents(
    chunks: list,
    file_name: str,
) -> list:
    """
    Converts chunk dictionaries into LangChain Documents,
    preserving retrieval and citation metadata.
    """

    documents = []

    optional_metadata_fields = [
        "source",
        "page_chunk_id",
        "paragraph_ids",
        "start_char",
        "end_char",
        "page_language",
        "page_language_confidence",
        "language",
        "language_confidence",
    ]

    for chunk in chunks:
        metadata = {
            "page_number": chunk["page_number"],
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "file_name": file_name,
        }

        for field in optional_metadata_fields:
            if field in chunk:
                metadata[field] = chunk[field]

        documents.append(
            Document(
                page_content=chunk["text"],
                metadata=metadata,
            )
        )

    return documents


# --------------------------------------------------
# VECTORSTORE DIRECTORY
# --------------------------------------------------

def _get_vectorstore_dir(
    user_id: int | None = None,
    document_id: str | None = None,
) -> Path:
    """
    Returns the vectorstore directory.

    With user_id:
        Uses isolated per-user storage.

    Without user_id:
        Uses the legacy global vectorstore directory.
    """

    if user_id is None:
        return VECTORSTORE_DIR
    if document_id:
        return get_user_document_vectorstore_dir(user_id, document_id)

    return get_user_vectorstore_dir(user_id)


def _resolve_vectorstore_dir(user_id: int | None = None, document_id: str | None = None) -> Path:
    """Keep older injected directory resolvers working during migration."""
    try:
        return _get_vectorstore_dir(user_id, document_id)
    except TypeError:
        return _get_vectorstore_dir(user_id)


@contextmanager
def vectorstore_read_lock(user_id: int | None = None, document_id: str | None = None):
    """Prevent reads from interleaving with transactional index replacement."""
    vectorstore_dir = _resolve_vectorstore_dir(user_id, document_id)
    vectorstore_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_file = vectorstore_dir.parent / ".vectorstore.lock"
    with lock_file.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def document_vectorstore_available(user_id: int, document_id: str) -> bool:
    """Return whether a complete, compatible per-document index exists."""
    vectorstore_dir = _resolve_vectorstore_dir(user_id, document_id)
    return (
        vectorstore_dir.is_dir()
        and _has_compatible_embedding_manifest(vectorstore_dir)
        and (vectorstore_dir / "index.faiss").is_file()
        and (vectorstore_dir / "index.pkl").is_file()
        and (vectorstore_dir / SPARSE_INDEX_FILENAME).is_file()
    )


def document_vectorstore_matches_identity(user_id: int, document_id: str, filename: str) -> bool:
    """Verify sparse metadata belongs to the requested document namespace."""
    vectorstore_dir = _resolve_vectorstore_dir(user_id, document_id)
    if not document_vectorstore_available(user_id, document_id):
        return False
    index = load_sparse_index(vectorstore_dir)
    if index is None or not index.records:
        return False
    return all(
        record.metadata.get("doc_id") == document_id
        and record.metadata.get("file_name") == filename
        for record in index.records
    )


# --------------------------------------------------
# CREATE AND SAVE VECTORSTORE
# --------------------------------------------------

def create_and_save_vectorstore(
    documents: list,
    user_id: int | None = None,
    document_id: str | None = None,
):
    """
    Creates and saves a FAISS vectorstore.

    Process:
        1. Generate embeddings.
        2. Create FAISS index.
        3. Save index to temporary directory.
        4. Write embedding manifest.
        5. Move existing index to backup.
        6. Move new index into place.
        7. Remove backup after successful replacement.

    If replacement fails, attempts to restore the old index.
    """

    if not documents:
        raise ValueError(
            "Cannot create vectorstore without documents."
        )

    embeddings = get_embedding_model()

    vectorstore = FAISS.from_documents(
        documents,
        embeddings,
    )

    vectorstore_dir = _resolve_vectorstore_dir(user_id, document_id)

    vectorstore_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=".vectorstore_tmp_",
            dir=str(vectorstore_dir.parent),
        )
    )

    try:
        vectorstore.save_local(str(temp_dir))
        _write_embedding_manifest(temp_dir)
        save_sparse_index(BM25Index.from_documents(documents), temp_dir)
        # The legacy API remains an immediately committed replacement.
        with VectorStoreTransaction(vectorstore_dir, temp_dir) as transaction:
            transaction.install()
            transaction.commit()
        return vectorstore
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


class VectorStoreTransaction:
    """A locked, reversible replacement of one vector-store directory."""

    def __init__(self, vectorstore_dir: Path, staged_dir: Path):
        self.vectorstore_dir = vectorstore_dir
        self.staged_dir = staged_dir
        self.backup_dir = None
        self.lock_file = None
        self.installed = False
        self.committed = False

    def __enter__(self):
        self.vectorstore_dir.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file = self.vectorstore_dir.parent / ".vectorstore.lock"
        self.lock_handle = self.lock_file.open("a+")
        fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX)
        return self

    def install(self):
        try:
            if self.vectorstore_dir.exists():
                self.backup_dir = Path(tempfile.mkdtemp(
                    prefix=".vectorstore_backup_",
                    dir=str(self.vectorstore_dir.parent),
                ))
                self.backup_dir.rmdir()
                self.vectorstore_dir.rename(self.backup_dir)
            self.staged_dir.rename(self.vectorstore_dir)
            self.installed = True
        except Exception:
            self.rollback()
            raise

    def mark_committed(self):
        """Mark the replacement committed without deleting its backup."""
        self.committed = True

    def commit(self):
        """Backward-compatible alias for the non-destructive commit mark."""
        self.mark_committed()

    def cleanup_backup(self):
        """Best-effort cleanup after all upload state is committed."""
        if self.backup_dir and self.backup_dir.exists():
            try:
                shutil.rmtree(self.backup_dir)
            except Exception:
                logger.exception(
                    "Unable to remove vectorstore backup %s",
                    self.backup_dir,
                )

    def rollback(self, *, force=False):
        if self.committed and not force:
            return
        if self.vectorstore_dir.exists() and self.installed:
            shutil.rmtree(self.vectorstore_dir, ignore_errors=True)
        if self.backup_dir and self.backup_dir.exists() and not self.vectorstore_dir.exists():
            self.backup_dir.rename(self.vectorstore_dir)
        if self.staged_dir.exists():
            shutil.rmtree(self.staged_dir, ignore_errors=True)
        self.installed = False
        self.backup_dir = None

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is not None or not self.committed:
                self.rollback(force=exc_type is not None)
            else:
                self.cleanup_backup()
        finally:
            if self.lock_handle:
                try:
                    fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_UN)
                finally:
                    self.lock_handle.close()
                    self.lock_handle = None
        return False


def begin_vectorstore_transaction(
    documents: list,
    user_id: int | None = None,
    document_id: str | None = None,
    progress_callback=None,
):
    """Build a staged vector store; the caller owns commit/rollback."""
    if not documents:
        raise ValueError("Cannot create vectorstore without documents.")
    vectorstore = FAISS.from_documents(
        documents,
        _embedding_adapter(get_embedding_model(), progress_callback),
    )
    vectorstore_dir = _resolve_vectorstore_dir(user_id, document_id)
    vectorstore_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(tempfile.mkdtemp(
        prefix=".vectorstore_tmp_", dir=str(vectorstore_dir.parent)
    ))
    try:
        vectorstore.save_local(str(staged_dir))
        _write_embedding_manifest(staged_dir)
        save_sparse_index(BM25Index.from_documents(documents), staged_dir)
    except Exception:
        shutil.rmtree(staged_dir, ignore_errors=True)
        raise
    transaction = VectorStoreTransaction(vectorstore_dir, staged_dir)
    transaction.vectorstore = vectorstore
    return transaction


# --------------------------------------------------
# LOAD VECTORSTORE
# --------------------------------------------------

def load_vectorstore(
    user_id: int | None = None,
    document_id: str | None = None,
):
    """
    Loads a FAISS vectorstore only when its embedding
    manifest matches the current embedding configuration.

    Old indexes without a manifest are rejected.
    """

    vectorstore_dir = _resolve_vectorstore_dir(user_id, document_id)

    with vectorstore_read_lock(user_id, document_id):
        if not vectorstore_dir.exists() or not _has_compatible_embedding_manifest(vectorstore_dir):
            return None
        try:
            files = [
                vectorstore_dir / "index.faiss",
                vectorstore_dir / "index.pkl",
                vectorstore_dir / EMBEDDING_MANIFEST_FILENAME,
                vectorstore_dir / SPARSE_INDEX_FILENAME,
            ]
            stats = tuple(
                (path.stat().st_ino, path.stat().st_mtime_ns, path.stat().st_size)
                for path in files
            )
            active = get_active_document(user_id) if user_id is not None else None
            document_id = document_id or (active or {}).get("active_doc_id")
            return _load_vectorstore_cached(
                user_id, document_id, stats, str(vectorstore_dir),
            )
        except Exception:
            return None


@lru_cache(maxsize=128)
def _load_vectorstore_cached(user_id: int | None, document_id: str | None, index_version: tuple, directory: str):
    """Reuse an immutable, explicitly user/document/version-scoped index."""
    del user_id, document_id, index_version
    return FAISS.load_local(
        directory,
        get_embedding_model(),
        allow_dangerous_deserialization=True,
    )


def load_sparse_vectorstore(user_id: int | None = None, document_id: str | None = None):
    """Load the sparse index from the same committed directory as FAISS."""
    vectorstore_dir = _resolve_vectorstore_dir(user_id, document_id)
    if not vectorstore_dir.exists() or not _has_compatible_embedding_manifest(vectorstore_dir):
        return None
    sparse_path = vectorstore_dir / SPARSE_INDEX_FILENAME
    try:
        stat = sparse_path.stat()
    except OSError:
        return None
    return _load_sparse_index_cached(
        str(vectorstore_dir), stat.st_mtime_ns, stat.st_size, stat.st_ino
    )


@lru_cache(maxsize=128)
def _load_sparse_index_cached(
    directory: str,
    modified_ns: int,
    size: int,
    inode: int,
) -> BM25Index | None:
    """Reuse an immutable committed BM25 index until its file changes."""
    # The metadata arguments are intentionally part of the cache key.  A
    # committed replacement gets a new mtime/size and therefore cannot reuse
    # stale records, while repeated queries avoid JSON parsing and BM25 setup.
    del modified_ns, size, inode
    return load_sparse_index(Path(directory))
