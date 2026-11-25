"""A small dense index. Embeddings and source metadata are saved locally."""

import hashlib
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import ollama

import storage

EMBED_MODEL = "bge-m3"
INDEX_FILE = Path(__file__).resolve().parent / "data" / "index.json"
BATCH_SIZE = 16


def current_documents():
    return [storage.read_document(path.stem) for path in sorted(storage.DATA.glob("*.json"))]


def fingerprint(documents):
    snapshot = [
        {"id": doc["id"], "text": doc["text"], "chunks": doc["chunks"]}
        for doc in sorted(documents, key=lambda doc: doc["id"])
    ]
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def unit_vector(values):
    if not isinstance(values, list) or not values:
        raise ValueError("An embedding is empty or invalid.")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError("An embedding contains invalid values.")
    magnitude = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(magnitude) or magnitude == 0:
        raise ValueError("An embedding has an invalid magnitude.")
    return [value / magnitude for value in values]


def embed_texts(texts):
    response = ollama.embed(model=EMBED_MODEL, input=texts, truncate=False)
    vectors = response.embeddings
    if len(vectors) != len(texts):
        raise ValueError("Ollama returned an unexpected number of embeddings.")
    return [unit_vector(vector) for vector in vectors]


def source_chunks(documents):
    records = []
    for document in documents:
        for chunk in document["chunks"]:
            if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
                raise ValueError("Invalid saved chunk. Recreate this document's chunks.")
            number, page = chunk.get("number"), chunk.get("page")
            if type(number) is not int or number < 1 or (page is not None and (type(page) is not int or page < 1)):
                raise ValueError("Invalid chunk location. Recreate the chunks.")
            records.append({
                "document_id": document["id"], "chunk": number,
                "page": page, "sheet": chunk.get("sheet"), "text": chunk["text"],
            })
    return records


def build_index():
    documents = current_documents()
    records = source_chunks(documents)
    if not records:
        raise ValueError("Create document chunks before building the index.")
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        for record, vector in zip(batch, embed_texts([record["text"] for record in batch])):
            record["embedding"] = vector
    dimensions = len(records[0]["embedding"])
    if any(len(record["embedding"]) != dimensions for record in records):
        raise ValueError("Embedding dimensions changed during indexing. Try again.")
    data = {
        "version": 1, "model": EMBED_MODEL, "dimensions": dimensions,
        "fingerprint": fingerprint(documents), "records": records,
    }
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with storage.write_lock():
        if fingerprint(current_documents()) != data["fingerprint"]:
            raise ValueError("Documents changed during indexing. Build again.")
        file = NamedTemporaryFile(mode="w", encoding="utf-8", dir=INDEX_FILE.parent, delete=False)
        temporary = Path(file.name)
        try:
            with file:
                json.dump(data, file, ensure_ascii=False, allow_nan=False)
            temporary.replace(INDEX_FILE)
        finally:
            temporary.unlink(missing_ok=True)
    return data


def load_index():
    try:
        data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError("Build the index on the Retrieval page first.") from error
    if not isinstance(data, dict) or data.get("version") != 1 or data.get("model") != EMBED_MODEL:
        raise ValueError("This index uses another format or embedding model. Rebuild it.")
    documents = current_documents()
    if data.get("fingerprint") != fingerprint(documents):
        raise ValueError("Documents changed. Rebuild the index before searching.")
    expected = source_chunks(documents)
    records = data.get("records")
    if not isinstance(records, list) or not records or len(records) != len(expected):
        raise ValueError("Invalid index records. Rebuild the index.")
    for record, source in zip(records, expected):
        if not isinstance(record, dict) or any(record.get(key) != value for key, value in source.items()):
            raise ValueError("Index sources do not match the saved chunks. Rebuild it.")
        vector = unit_vector(record.get("embedding"))
        if len(vector) != data.get("dimensions"):
            raise ValueError("Invalid embedding dimensions. Rebuild the index.")
    return data, {document["id"]: document for document in documents}
