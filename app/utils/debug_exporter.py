import json
import logging
import shutil
import tempfile
from pathlib import Path
from app.core.config import DEBUG_OUTPUT_DIR
from app.services.user_storage_service import get_user_debug_dir


logger = logging.getLogger(__name__)


class DebugExportTransaction:
    def __init__(self, base_dir, staged_dir):
        self.base_dir = base_dir
        self.staged_dir = staged_dir
        self.backup_dir = None
        self.installed = False
        self.committed = False

    def install(self):
        """Install staged output while retaining the previous output."""
        if self.installed:
            return
        try:
            if self.base_dir.exists():
                backup_parent = self.base_dir.parent
                self.backup_dir = Path(tempfile.mkdtemp(
                    prefix=f".{self.base_dir.name}.debug_backup_",
                    dir=str(backup_parent),
                ))
                self.backup_dir.rmdir()
                self.base_dir.rename(self.backup_dir)
            self.staged_dir.rename(self.base_dir)
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
                    "Unable to remove debug backup %s", self.backup_dir
                )

    def rollback(self, *, force=False):
        if self.committed and not force:
            return
        if self.base_dir.exists() and self.installed:
            shutil.rmtree(self.base_dir, ignore_errors=True)
        if self.backup_dir and self.backup_dir.exists() and not self.base_dir.exists():
            self.backup_dir.rename(self.base_dir)
        if self.staged_dir.exists():
            shutil.rmtree(self.staged_dir, ignore_errors=True)
        self.installed = False
        self.backup_dir = None


def export_debug_files(source_filename, pages_data, metadata, user_id: int | None = None, *, commit=True):
    safe_name = Path(source_filename).stem

    if user_id is not None:
        base_dir = get_user_debug_dir(user_id) / safe_name
    else:
        base_dir = DEBUG_OUTPUT_DIR / safe_name

    staging_parent = base_dir.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(tempfile.mkdtemp(prefix=f".{safe_name}.debug_tmp_", dir=str(staging_parent)))
    clean_pages_dir = staged_dir / "clean_pages"
    chunks_dir = staged_dir / "chunks"

    clean_pages_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save clean extracted pages
        for page in pages_data:
            page_num = page["page_number"]
            page_text = page["text"]
            page_file = clean_pages_dir / f"page_{page_num:03}.txt"
            with open(page_file, "w", encoding="utf-8") as f:
                f.write(page_text)

        # Save chunks separately
        for item in metadata:
            page_num = item["page_number"]
            page_chunk_id = item["page_chunk_id"]
            chunk_text = item["text"]
            chunk_file = chunks_dir / f"page_{page_num:03}_chunk_{page_chunk_id:02}.txt"
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(chunk_text)

        metadata_file = staged_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
    except Exception:
        shutil.rmtree(staged_dir, ignore_errors=True)
        raise

    transaction = DebugExportTransaction(base_dir, staged_dir)
    if commit:
        try:
            transaction.install()
            transaction.mark_committed()
            transaction.cleanup_backup()
        except Exception:
            transaction.rollback()
            raise
        return None
    return transaction
