from app.utils import debug_exporter


def test_debug_export_replaces_stale_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(debug_exporter, "DEBUG_OUTPUT_DIR", tmp_path)
    debug_exporter.export_debug_files(
        "report.pdf",
        [{"page_number": 1, "text": "page"}],
        [{"page_number": 1, "page_chunk_id": 1, "text": "one"},
         {"page_number": 1, "page_chunk_id": 2, "text": "two"}],
    )
    debug_exporter.export_debug_files(
        "report.pdf",
        [{"page_number": 1, "text": "page"}],
        [{"page_number": 1, "page_chunk_id": 1, "text": "one"}],
    )

    chunks = tmp_path / "report" / "chunks"
    assert (chunks / "page_001_chunk_01.txt").exists()
    assert not (chunks / "page_001_chunk_02.txt").exists()


def test_debug_export_preserves_old_output_if_staging_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(debug_exporter, "DEBUG_OUTPUT_DIR", tmp_path)
    debug_exporter.export_debug_files(
        "report.pdf",
        [{"page_number": 1, "text": "old"}],
        [{"page_number": 1, "page_chunk_id": 1, "text": "old"}],
    )

    try:
        debug_exporter.export_debug_files(
            "report.pdf", [{"page_number": 1, "text": "new"}], [{}]
        )
    except KeyError:
        pass
    else:
        raise AssertionError("Expected staged export failure")

    assert (
        (tmp_path / "report" / "clean_pages" / "page_001.txt")
        .read_text()
        == "old"
    )


def test_debug_transaction_install_retains_backup_until_cleanup(tmp_path):
    base_dir = tmp_path / "report"
    base_dir.mkdir()
    (base_dir / "marker").write_text("old")
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    (staged_dir / "marker").write_text("new")

    transaction = debug_exporter.DebugExportTransaction(
        base_dir, staged_dir
    )
    transaction.install()

    assert (base_dir / "marker").read_text() == "new"
    assert transaction.backup_dir.exists()

    transaction.mark_committed()
    transaction.cleanup_backup()
    assert not transaction.backup_dir.exists()


def test_debug_transaction_cleanup_failure_does_not_raise(tmp_path, monkeypatch):
    base_dir = tmp_path / "report"
    base_dir.mkdir()
    (base_dir / "marker").write_text("old")
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()

    transaction = debug_exporter.DebugExportTransaction(
        base_dir, staged_dir
    )
    transaction.install()
    transaction.mark_committed()

    def fail_rmtree(path, *args, **kwargs):
        raise OSError("cleanup failure")

    monkeypatch.setattr(debug_exporter.shutil, "rmtree", fail_rmtree)
    transaction.cleanup_backup()
    assert transaction.committed
