"""Document text extraction and overlapping word chunks."""

from io import BytesIO
from pathlib import Path


def extract_text(name, content):
    suffix = Path(name).suffix.lower()
    if suffix == ".txt":
        text = content.decode("utf-8-sig")
    elif suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(content))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix == ".docx":
        from docx import Document
        document = Document(BytesIO(content))
        # XML order keeps paragraphs and table cells in reading order.
        text = "\n".join(node.text for node in document.element.body.iter()
                         if node.tag.endswith("}t") and node.text)
    elif suffix == ".hwp":
        text = hwp_text(content)
    elif suffix == ".hwpx":
        text = hwpx_text(content)
    else:
        raise ValueError("Choose a TXT, PDF, DOCX, HWP, or HWPX file.")
    text = text.strip()
    if not text:
        raise ValueError("No readable text found. Scanned PDFs need OCR before upload.")
    return text


def chunk_text(text, size=150, overlap=30):
    if not 0 <= overlap < size:
        raise ValueError("Overlap must be smaller than the chunk size.")
    words = text.split()
    chunks = []
    for start in range(0, len(words), size - overlap):
        chunks.append({
            "number": len(chunks) + 1,
            "start_word": start,
            "end_word": min(start + size, len(words)),
            "text": " ".join(words[start:start + size]),
        })
        if start + size >= len(words):
            break
    return chunks


# HWP 5 body records and HWPX section XML are parsed as plain text.
MAX_EXPANDED_BYTES = 30 * 1024 * 1024


def hwpx_text(content):
    import re
    from zipfile import ZipFile
    from xml.etree import ElementTree

    class TextTreeBuilder(ElementTree.TreeBuilder):
        def doctype(self, name, pubid, system):
            raise ValueError("HWPX sections with XML document types are unsupported.")

    paragraphs = []
    with ZipFile(BytesIO(content)) as archive:
        sections = []
        for entry in archive.infolist():
            match = re.fullmatch(r"Contents/section(\d+)\.xml", entry.filename)
            if match:
                sections.append((int(match[1]), entry))
        if not sections:
            raise ValueError("The HWPX file has no readable body sections.")
        if sum(entry.file_size for _, entry in sections) > MAX_EXPANDED_BYTES:
            raise ValueError("Expanded HWPX text sections exceed the classroom limit.")
        for _, entry in sorted(sections, key=lambda item: item[0]):
            root = ElementTree.fromstring(archive.read(entry), parser=ElementTree.XMLParser(target=TextTreeBuilder()))
            # Read text runs once in XML order, including nested table paragraphs.
            for node in root.iter():
                local = node.tag.rsplit("}", 1)[-1]
                if local == "t":
                    paragraphs.append("".join(node.itertext()))
                elif local == "p":
                    paragraphs.append("\n")
    return "".join(paragraphs)


def hwp_paragraph(payload):
    import struct

    if len(payload) % 2:
        raise ValueError("Damaged HWP paragraph.")
    units = struct.unpack(f"<{len(payload) // 2}H", payload)
    output = bytearray()
    index = 0
    while index < len(units):
        code = units[index]
        if code < 32:
            if code in (9, 10, 13, 30, 31):
                output.extend((" " if code in (9, 30, 31) else "\n").encode("utf-16le"))
            width = 8 if code in (*range(1, 10), 11, 12, *range(14, 24)) else 1
            if index + width > len(units):
                raise ValueError("Damaged HWP control.")
            index += width
        else:
            output.extend(struct.pack("<H", code))
            index += 1
    return output.decode("utf-16le")


def hwp_section_text(data):
    import struct

    paragraphs = []
    offset = 0
    while offset < len(data):
        if offset + 4 > len(data):
            raise ValueError("Truncated HWP record header.")
        header = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        tag, size = header & 1023, header >> 20
        if size == 4095:
            if offset + 4 > len(data):
                raise ValueError("Truncated HWP record length.")
            size = struct.unpack_from("<I", data, offset)[0]
            offset += 4
        if offset + size > len(data):
            raise ValueError("Truncated HWP record.")
        if tag == 67:
            paragraphs.append(hwp_paragraph(data[offset:offset + size]))
        offset += size
    return "\n".join(paragraphs)


def hwp_text(content):
    import re
    import struct
    import zlib
    try:
        import olefile
    except ImportError as error:
        raise ValueError("HWP extraction needs olefile. Install requirements.txt again.") from error

    paragraphs = []
    with olefile.OleFileIO(BytesIO(content)) as document:
        header = document.openstream("FileHeader").read()
        if len(header) < 40 or header[:32] != b"HWP Document File" + b"\0" * 15 or header[35] != 5:
            raise ValueError("Only HWP 5 documents are supported. Save older documents as HWPX.")
        flags = struct.unpack_from("<I", header, 36)[0]
        if flags & ((1 << 1) | (1 << 2) | (1 << 4) | (1 << 8) | (1 << 10)):
            raise ValueError("Protected or distribution HWP documents need an unprotected HWPX export.")
        sections = [
            path for path in document.listdir()
            if len(path) == 2 and path[0] == "BodyText" and re.fullmatch(r"Section\d+", path[1])
        ]
        if not sections:
            raise ValueError("The HWP file has no readable body sections.")
        remaining = MAX_EXPANDED_BYTES
        for path in sorted(sections, key=lambda path: int(path[1][7:])):
            raw = document.openstream(path).read(remaining + 1)
            if flags & 1:
                decoder = zlib.decompressobj(-15)
                raw = decoder.decompress(raw, remaining + 1)
                if not decoder.eof:
                    raise ValueError("HWP section exceeds the extraction limit or is damaged.")
            remaining -= len(raw)
            if remaining < 0:
                raise ValueError("Expanded HWP sections exceed the classroom limit.")
            paragraphs.append(hwp_section_text(raw))
    return "\n".join(paragraphs)

