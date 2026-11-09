
from app.services.chunk_service import chunk_page_text_with_metadata
from app.services.language_detection_service import detect_text_language


def build_chunk_metadata(pages_data, source_filename, doc_id):
    metadata = []
    chunk_counter = 1
    chunk_configs = []

    for page in pages_data:
        page_number = page["page_number"]
        page_text = page["text"]

        if not page_text or not page_text.strip():
            continue

        # Detect dominant script/language for the page.
        page_language_info = detect_text_language(page_text)

        page_chunks, config = chunk_page_text_with_metadata(page_text)
        chunk_configs.append(config)

        for page_chunk_id, chunk in enumerate(page_chunks, start=1):

            # Detect script/language for each chunk.
            chunk_language_info = detect_text_language(chunk["text"])

            metadata.append({
                # Existing fields.
                "chunk_id": chunk_counter,
                "page_chunk_id": page_chunk_id,
                "doc_id": doc_id,
                "text": chunk["text"],
                "page_number": page_number,
                "source": source_filename,

                # Paragraph and character-position metadata.
                "paragraph_ids": chunk["paragraph_ids"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],

                # Language metadata.
                "page_language": page_language_info["language"],
                "page_language_confidence": (
                    page_language_info["confidence"]
                ),
                "language": chunk_language_info["language"],
                "language_confidence": (
                    chunk_language_info["confidence"]
                ),
            })

            chunk_counter += 1

    return metadata, chunk_configs
