"""One atomic JSON record per document, including its original bytes."""

import base64
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from uuid import UUID, uuid4

import streamlit as st

DATA = Path(__file__).resolve().parent / "data" / "documents"
MAX_BYTES = 10 * 1024 * 1024


@st.cache_resource
def write_lock():
    return Lock()


def record_path(identifier):
    return DATA / f"{UUID(identifier)}.json"


def read_document(identifier):
    data = json.loads(record_path(identifier).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("id") != identifier:
        raise ValueError("Invalid document record.")
    if not isinstance(data.get("name"), str) or not isinstance(data.get("original_name"), str):
        raise ValueError("Invalid document name.")
    if not isinstance(data.get("text"), str) or not isinstance(data.get("chunks"), list):
        raise ValueError("Invalid processed content.")
    base64.b64decode(data["content"], validate=True)
    return data


def write_document(document, previous=None):
    DATA.mkdir(parents=True, exist_ok=True)
    path = record_path(document["id"])
    with write_lock():
        if previous is not None and read_document(document["id"]) != previous:
            raise ValueError("This document changed in another tab. Refresh and try again.")
        file = NamedTemporaryFile(mode="w", encoding="utf-8", dir=DATA, delete=False)
        temporary = Path(file.name)
        try:
            with file:
                json.dump(document, file, ensure_ascii=False, indent=2)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def upload_document(name, content):
    name = name.replace("\\", "/").split("/")[-1].strip()
    if Path(name).suffix.lower() not in {".txt", ".pdf", ".docx", ".hwp", ".hwpx"}:
        raise ValueError("Choose a TXT, PDF, DOCX, HWP, or HWPX file.")
    if not content or len(content) > MAX_BYTES:
        raise ValueError("Choose a nonempty file of at most 10 MB.")
    document = {
        "id": str(uuid4()), "name": name, "original_name": name,
        "content": base64.b64encode(content).decode("ascii"),
        "text": "", "chunks": [], "chunk_size": None, "overlap": None,
    }
    write_document(document)
    return document


def rename_document(previous, name):
    name = name.strip()
    if not name or len(name) > 120 or any(char in name for char in "/\\\n\r"):
        raise ValueError("Use a name of 1–120 characters without slashes or line breaks.")
    updated = {**previous, "name": name}
    write_document(updated, previous)
    return updated


def delete_document(previous):
    with write_lock():
        if read_document(previous["id"]) != previous:
            raise ValueError("This document changed in another tab. Refresh and try again.")
        record_path(previous["id"]).unlink()
