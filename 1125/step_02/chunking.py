"""Reuse word chunking and attach source-page metadata."""

from pathlib import Path

from processing import chunk_text


def chunk_document(document, size=150, overlap=30):
    pages = document.get("pages")
    if pages is None:
        if Path(document["original_name"]).suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".xlsx"}:
            raise ValueError("Extract this document again to recover page or worksheet references.")
        pages = [{"page": None, "text": document["text"], "method": "Legacy text"}]

    chunks = []
    for page in pages:
        for chunk in chunk_text(page["text"], size, overlap):
            if page.get("sheet"):
                chunk["text"] = f"Worksheet: {page['sheet']}\n" + chunk["text"]
            chunks.append({
                **chunk, "number": len(chunks) + 1,
                "page": page["page"], "sheet": page.get("sheet"), "document_id": document["id"],
            })
    if not chunks:
        raise ValueError("Extract readable text before creating chunks.")
    return chunks
