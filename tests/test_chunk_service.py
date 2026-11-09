
from app.services.chunk_service import (
    chunk_page_text,
    chunk_page_text_with_metadata,
)
from app.utils.metadata_builder import build_chunk_metadata


def test_empty_text_returns_no_chunks():
    chunks, config = chunk_page_text("   ")

    assert chunks == []
    assert config == {
        "chunk_size": 0,
        "chunk_overlap": 0,
    }


def test_backward_compatible_chunk_interface():
    text = "This is a test paragraph. " * 30

    chunks, config = chunk_page_text(text)

    assert chunks
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert config["chunk_size"] > 0
    assert config["chunk_overlap"] >= 0


def test_chunk_metadata_contains_valid_character_offsets():
    text = (
        "First paragraph contains useful information.\n\n"
        "Second paragraph contains additional information."
    )

    chunks, _ = chunk_page_text_with_metadata(text)

    assert chunks

    for chunk in chunks:
        start = chunk["start_char"]
        end = chunk["end_char"]

        assert start >= 0
        assert end > start
        assert text[start:end] == chunk["text"]
        assert chunk["paragraph_ids"]


def test_chunk_metadata_identifies_paragraphs_correctly():
    text = (
        "First paragraph contains useful information.\n\n"
        "Second paragraph contains additional information."
    )

    chunks, _ = chunk_page_text_with_metadata(text)

    assert chunks

    first_paragraph_chunks = [
        chunk for chunk in chunks
        if 1 in chunk["paragraph_ids"]
    ]

    second_paragraph_chunks = [
        chunk for chunk in chunks
        if 2 in chunk["paragraph_ids"]
    ]

    assert first_paragraph_chunks
    assert second_paragraph_chunks


def test_metadata_builder_preserves_existing_fields():
    pages = [
        {
            "page_number": 1,
            "text": "First paragraph.\n\nSecond paragraph.",
        }
    ]

    metadata, configs = build_chunk_metadata(
        pages,
        "sample.pdf",
        "doc-123",
    )

    assert metadata
    assert configs

    chunk = metadata[0]

    assert chunk["chunk_id"] == 1
    assert chunk["page_chunk_id"] == 1
    assert chunk["doc_id"] == "doc-123"
    assert chunk["page_number"] == 1
    assert chunk["source"] == "sample.pdf"

    assert "paragraph_ids" in chunk
    assert "start_char" in chunk
    assert "end_char" in chunk


def test_chunks_do_not_cross_page_boundaries():
    pages = [
        {
            "page_number": 1,
            "text": "Page one content. " * 30,
        },
        {
            "page_number": 2,
            "text": "Page two content. " * 30,
        },
    ]

    metadata, _ = build_chunk_metadata(
        pages,
        "sample.pdf",
        "doc-123",
    )

    assert metadata

    page_numbers = {
        chunk["page_number"] for chunk in metadata
    }

    assert page_numbers == {1, 2}

    for chunk in metadata:
        if chunk["page_number"] == 1:
            assert "Page one content." in chunk["text"]
            assert "Page two content." not in chunk["text"]

        elif chunk["page_number"] == 2:
            assert "Page two content." in chunk["text"]
            assert "Page one content." not in chunk["text"]


def test_metadata_builder_detects_kannada_language():
    pages = [{
        "page_number": 1,
        "text": "ಕರ್ನಾಟಕ ಸರ್ಕಾರ ಗ್ರಾಮ ಪಂಚಾಯಿತಿ ಅಭಿವೃದ್ಧಿ",
    }]

    metadata, _ = build_chunk_metadata(
        pages,
        "kannada.pdf",
        "doc-kn",
    )

    assert metadata
    assert metadata[0]["language"] == "kn"
    assert metadata[0]["page_language"] == "kn"


def test_metadata_builder_detects_tamil_language():
    pages = [{
        "page_number": 1,
        "text": "தமிழ்நாடு அரசு ஊராட்சி வளர்ச்சி",
    }]

    metadata, _ = build_chunk_metadata(
        pages,
        "tamil.pdf",
        "doc-ta",
    )

    assert metadata
    assert metadata[0]["language"] == "ta"
    assert metadata[0]["page_language"] == "ta"


def test_metadata_builder_detects_telugu_language():
    pages = [{
        "page_number": 1,
        "text": "ఆంధ్రప్రదేశ్ ప్రభుత్వం గ్రామ పంచాయతీ",
    }]

    metadata, _ = build_chunk_metadata(
        pages,
        "telugu.pdf",
        "doc-te",
    )

    assert metadata
    assert metadata[0]["language"] == "te"
    assert metadata[0]["page_language"] == "te"


def test_metadata_builder_detects_mixed_page_language():
    pages = [{
        "page_number": 1,
        "text": (
            "ಕರ್ನಾಟಕ ಸರ್ಕಾರ Karnataka Government"
        ),
    }]

    metadata, _ = build_chunk_metadata(
        pages,
        "mixed.pdf",
        "doc-mixed",
    )

    assert metadata
    assert all(
        chunk["page_language"] == "mixed"
        for chunk in metadata
    )

    assert all(
        chunk["language"] in {"kn", "en", "mixed"}
        for chunk in metadata
    )
