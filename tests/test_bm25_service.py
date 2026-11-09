from langchain_core.documents import Document

from app.services.bm25_service import BM25Index, load_sparse_index, save_sparse_index, tokenize


def test_multilingual_tokenization_keeps_letters_marks_numbers_and_identifiers():
    samples = [
        "English medical diagnosis 2024-A1",
        "हिन्दी रक्तचाप 120/80",
        "ಕನ್ನಡ ಹೃದಯ 2024",
        "தமிழ் இரத்த அழுத்தம்",
        "తెలుగు రక్తపోటు",
        "മലയാളം രക്തസമ്മർദ്ദം",
        "বাংলা রক্তচাপ",
        "ગુજરાતી રક્તચાપ",
        "मराठी रक्तदाब",
        "ਪੰਜਾਬੀ ਬਲੱਡ ਪ੍ਰੈਸ਼ਰ",
        "اردو بلڈ پریشر",
    ]
    for sample in samples:
        tokens = tokenize(sample)
        assert tokens
        assert any(any(char.isalpha() for char in token) for token in tokens)
    assert "2024" in tokenize("2024-A1")
    assert "a1" in tokenize("2024-A1")
    assert tokenize("किताब") == tokenize("किताब")


def test_empty_punctuation_mixed_and_rtl_are_safe():
    assert tokenize("") == []
    assert tokenize("!!! — ...") == []
    assert tokenize("اردو English 42-A") == ["اردو", "english", "42", "a"]


def test_sparse_exact_term_retrieval_and_json_round_trip(tmp_path):
    index = BM25Index.from_documents([
        Document(page_content="कन्नड़ चिकित्सा शब्दावली", metadata={"doc_id": "d", "chunk_id": 1}),
        Document(page_content="Urdu بلڈ پریشر identifier X-99", metadata={"doc_id": "d", "chunk_id": 2}),
    ])
    assert index.search("X-99", 2)[0]["metadata"]["chunk_id"] == 2
    save_sparse_index(index, tmp_path)
    restored = load_sparse_index(tmp_path)
    assert restored is not None
    assert restored.search("चिकित्सा", 2)[0]["metadata"]["chunk_id"] == 1
    
    

def test_from_documents_supports_empty_langchain_document():
    document = Document(
        page_content="",
        metadata={},
    )

    index = BM25Index.from_documents([document])

    assert len(index.records) == 1
    assert index.records[0].text == ""
    assert index.records[0].tokens == []


def test_from_documents_supports_dictionary():
    index = BM25Index.from_documents([
        {
            "page_content": "ಕನ್ನಡ ದಾಖಲೆ",
            "metadata": {"doc_id": "doc-1", "chunk_id": 1},
        }
    ])

    assert len(index.records) == 1
    assert index.records[0].metadata["doc_id"] == "doc-1"


def test_search_rejects_negative_limit():
    index = BM25Index.from_documents([
        Document(page_content="medical diagnosis", metadata={})
    ])

    import pytest

    with pytest.raises(ValueError):
        index.search("medical", limit=-1)


def test_search_zero_limit_returns_empty():
    index = BM25Index.from_documents([
        Document(page_content="medical diagnosis", metadata={})
    ])

    assert index.search("medical", limit=0) == []


def test_sparse_index_load_rejects_invalid_record(tmp_path):
    import json

    path = tmp_path / "sparse_index.json"

    path.write_text(
        json.dumps({
            "schema_version": 1,
            "records": [
                {
                    "text": "medical document",
                    "metadata": {},
                    "tokens": ["incorrect-token"],
                }
            ],
        }),
        encoding="utf-8",
    )

    assert load_sparse_index(tmp_path) is None

