
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_page_chunk_config(text: str):
    total_chars = len(text)

    if total_chars < 800:
        chunk_size = 300
        chunk_overlap = 50
    elif total_chars < 2000:
        chunk_size = 400
        chunk_overlap = 80
    elif total_chars < 5000:
        chunk_size = 500
        chunk_overlap = 100
    else:
        chunk_size = 700
        chunk_overlap = 120

    if total_chars < chunk_size:
        chunk_size = max(total_chars, 50)

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(chunk_size // 5, 10)

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def chunk_page_text_with_metadata(text: str):
    """
    Split one PDF page while preserving paragraph boundaries
    where possible.

    Character offsets refer to the original extracted page text.
    """

    if not text or not text.strip():
        return [], {"chunk_size": 0, "chunk_overlap": 0}

    leading_whitespace = len(text) - len(text.lstrip())
    cleaned_text = text.strip()

    config = get_page_chunk_config(cleaned_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )

    split_documents = splitter.create_documents([cleaned_text])

    # Find paragraph spans in the original extracted page text.
    paragraph_spans = []

    for paragraph_id, match in enumerate(
        re.finditer(r"\S.*?(?=\n\s*\n|\Z)", cleaned_text, re.DOTALL),
        start=1,
    ):
        paragraph_spans.append(
            (paragraph_id, match.start(), match.end())
        )

    chunks = []

    for document in split_documents:
        chunk_text = document.page_content
        relative_start = document.metadata.get("start_index")

        if relative_start is None:
            raise ValueError(
                "Text splitter did not return a chunk start index."
            )

        relative_end = relative_start + len(chunk_text)

        paragraph_ids = [
            paragraph_id
            for paragraph_id, paragraph_start, paragraph_end
            in paragraph_spans
            if (
                paragraph_start < relative_end
                and paragraph_end > relative_start
            )
        ]

        chunks.append({
            "text": chunk_text,
            "start_char": leading_whitespace + relative_start,
            "end_char": leading_whitespace + relative_end,
            "paragraph_ids": paragraph_ids,
        })

    return chunks, config


def chunk_page_text(text: str):
    """
    Backward-compatible interface.

    Existing callers still receive (list[str], config).
    """

    chunks, config = chunk_page_text_with_metadata(text)

    return [chunk["text"] for chunk in chunks], config
