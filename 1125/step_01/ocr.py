"""English/Korean OCR with page-aware extraction."""

from io import BytesIO
from pathlib import Path

import streamlit as st

from processing import extract_text

MAX_PAGES = 50
MAX_PIXELS = 12_000_000


@st.cache_resource
def ocr_reader():
    import easyocr

    model_directory = Path(__file__).resolve().parent / "data" / "ocr_models"
    model_directory.mkdir(parents=True, exist_ok=True)
    return easyocr.Reader(["ko", "en"], gpu=False, model_storage_directory=str(model_directory))


def recognize_image(content):
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(BytesIO(content)) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("Use an image smaller than 12 megapixels.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        lines = ocr_reader().readtext(np.asarray(image), detail=0, paragraph=True)
    return "\n".join(lines).strip()


def extract_pages(name, content, force_ocr=False):
    suffix = Path(name).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg"}:
        pages = [{"page": 1, "text": recognize_image(content), "method": "OCR"}]
    elif suffix == ".xlsx":
        from spreadsheet import extract_sheets
        pages = extract_sheets(content)
    elif suffix == ".pdf":
        import pymupdf

        pages = []
        with pymupdf.open(stream=content, filetype="pdf") as document:
            if document.needs_pass:
                raise ValueError("Open an unprotected PDF export.")
            if len(document) > MAX_PAGES:
                raise ValueError("Use a PDF with at most 50 pages for this workshop.")
            for number, page in enumerate(document, 1):
                text = page.get_text("text").strip()
                method = "PDF text"
                if force_ocr or not text:
                    # Bound the render before allocating its pixel buffer.
                    scale = min(2.0, (MAX_PIXELS / (page.rect.width * page.rect.height)) ** 0.5)
                    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                    text = recognize_image(pixmap.tobytes("png"))
                    method = "OCR"
                pages.append({"page": number, "text": text, "method": method})
    else:
        # Office/plain-text readers do not provide dependable physical page numbers.
        pages = [{"page": None, "text": extract_text(name, content), "method": "Native text"}]
    if not any(page["text"].strip() for page in pages):
        raise ValueError("No text found. Try a clearer image or an unprotected text document.")
    return pages
