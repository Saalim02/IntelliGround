import json

import pytest

from benchmarks import multilingual_corpus_builder as builder


def _fake_extraction(path, language_hint):
    return {
        "total_pages": 2,
        "pages": [
            {
                "page_number": 1,
                "text": "বাংলা সংবাদ",
                "extraction_method": "native",
                "ocr_language": "unused-on-native",
            },
            {
                "page_number": 2,
                "text": "",
                "extraction_method": "blank",
            },
        ],
    }


def test_build_corpus_writes_isolated_utf8_artifacts(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source"
    source.mkdir()
    for filename, _hint in builder.PDFS.values():
        (source / filename).write_bytes(filename.encode())

    monkeypatch.setattr(builder, "extract_text_from_pdf", _fake_extraction)
    output = tmp_path / "benchmark"
    report = builder.build_corpus(source, output)

    assert report["successful_pdf_count"] == 10
    assert report["error_pdf_count"] == 0
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["corpus_size"] == 10

    pages = json.loads((output / "pages_bengali.json").read_text(encoding="utf-8"))
    assert pages["pages"][0]["text"] == "বাংলা সংবাদ"
    assert pages["pages"][0]["ocr_language"] == "unused-on-native"
    assert pages["pages"][1]["page_number"] == 2
    assert pages["statistics"]["extraction_method_counts"] == {
        "native": 1,
        "blank": 1,
    }

    chunks = json.loads((output / "chunks_bengali.json").read_text(encoding="utf-8"))
    assert chunks["chunks"][0]["doc_id"].startswith("benchmark-multilingual_20261105-bengali-")
    assert chunks["documents"][0]["metadata"]["doc_id"] == chunks["chunks"][0]["doc_id"]
    progress = capsys.readouterr().out
    assert progress.count("status=success") == 10
    assert "pages=2 chunks=1 extraction_methods=blank=1, native=1" in progress
    assert "elapsed_seconds=" in progress


def test_missing_source_is_reported_as_error(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "extract_text_from_pdf", pytest.fail)
    output = tmp_path / "benchmark"
    report = builder.build_corpus(tmp_path / "missing", output)

    assert report["error_pdf_count"] == 10
    assert report["pdfs"][0]["status"] == "error"
    assert report["pdfs"][0]["source"]["sha256"] is None
    pages = json.loads((output / "pages_bengali.json").read_text(encoding="utf-8"))
    assert pages["errors"]


def test_existing_output_directory_is_never_modified(tmp_path):
    output = tmp_path / "benchmark"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        builder.build_corpus(tmp_path, output)

    assert marker.read_text(encoding="utf-8") == "preserve"
