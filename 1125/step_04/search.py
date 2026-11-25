"""Cosine ranking and a visible search experiment."""

import streamlit as st

from indexing import embed_texts, load_index, unit_vector


def search(query, limit=3, minimum_score=0.3):
    if not query.strip():
        raise ValueError("Enter a question to search.")
    index, documents = load_index()
    vector = embed_texts([query])[0]
    if len(vector) != index["dimensions"]:
        raise ValueError("The query and index use different dimensions. Rebuild the index.")
    results = []
    for record in index["records"]:
        score = sum(a * b for a, b in zip(vector, unit_vector(record["embedding"])))
        if score >= minimum_score:
            results.append({**record, "name": documents[record["document_id"]]["name"], "score": score})
    return sorted(results, key=lambda result: result["score"], reverse=True)[:limit]


def show_results(results):
    for number, result in enumerate(results, 1):
        labels = [f"[{number}] {result['name']}"]
        if result.get("sheet"):
            labels.append(f"sheet {result['sheet']}")
        elif result.get("page") is not None:
            labels.append(f"page {result['page']}")
        labels.append(f"chunk {result['chunk']}")
        with st.expander(" · ".join(labels), expanded=True):
            st.caption(f"Cosine similarity {result['score']:.3f} · Document {result['document_id']}")
            st.text(result["text"])
