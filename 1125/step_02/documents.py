"""Document page. Checkpoint number controls which lesson features appear."""

import base64

import streamlit as st

import storage


def document_page(step):
    st.title("Your document library")
    st.write("Give your materials a home. Prepare their text for the retrieval workshop.")
    message = st.session_state.pop("document_notice", None)
    if message:
        st.success(message)

    with st.form("upload", clear_on_submit=True):
        file = st.file_uploader("Add a document", type=["txt", "pdf", "docx", "hwp", "hwpx", "xlsx", "png", "jpg", "jpeg"])
        submitted = st.form_submit_button("Upload document", type="primary")
    if submitted:
        if file is None:
            st.warning("Choose a file first.")
        else:
            try:
                storage.upload_document(file.name, file.getvalue())
            except (OSError, ValueError) as error:
                st.error(str(error))
            else:
                st.session_state.document_notice = "Document uploaded."
                st.rerun()

    documents = []
    for path in sorted(storage.DATA.glob("*.json")):
        try:
            documents.append(storage.read_document(path.stem))
        except (OSError, ValueError, KeyError, TypeError) as error:
            st.error(f"Could not read {path.name}. {error}")
    if not documents:
        st.info("Your library is ready for its first document. Upload a document or a PNG/JPEG image.")
        return

    options = {document["id"]: document for document in documents}
    identifier = st.selectbox(
        "Choose a document", list(options),
        format_func=lambda key: f"{options[key]['name']} · {key[:8]}",
    )
    document = options[identifier]
    st.subheader(document["name"])
    st.caption(f"Original file: {document['original_name']} · Reference {identifier[:8]}")
    st.download_button(
        "Download original", base64.b64decode(document["content"]),
        file_name=document["original_name"],
    )

    if step >= 3:
        with st.expander("Manage document"):
            with st.form("rename_" + identifier):
                name = st.text_input("Display name", value=document["name"], max_chars=120)
                renamed = st.form_submit_button("Save name")
            if renamed:
                mutate(lambda: storage.rename_document(document, name), "Document renamed.")
            confirmed = st.checkbox("Delete this document and its extracted text and chunks", key=identifier)
            if st.button("Delete document", disabled=not confirmed):
                mutate(lambda: storage.delete_document(document), "Document deleted.")

    if step >= 4:
        from ocr import extract_pages

        force_ocr = st.checkbox(
            "OCR every PDF page", help="Use for mixed text/image pages or a faulty PDF text layer."
        )
        st.caption("OCR reads Korean and English. The first use downloads the recognition models.")

        if st.button("Extract text", type="primary"):
            def extract():
                with st.spinner("Reading the document…"):
                    pages = extract_pages(
                        document["original_name"], base64.b64decode(document["content"]), force_ocr,
                    )
                    text = "\n\n".join(page["text"] for page in pages)
                storage.write_document(
                    {**document, "text": text, "pages": pages, "chunks": [], "chunk_size": None, "overlap": None},
                    document,
                )
            mutate(extract, "Text extracted. Create chunks from this version of the text.")
        if document["text"]:
            st.caption(f"{len(document['text'].split()):,} words extracted")
            with st.expander("Extracted text", expanded=True):
                st.text(document["text"])
            for page in document.get("pages", []):
                location = f"Sheet {page['sheet']}" if page.get("sheet") else (f"Page {page['page']}" if page["page"] is not None else "Document text")
                with st.expander(f"{location} · {page['method']}"):
                    st.text(page["text"] or "(No readable text on this page)")


    if step >= 5 and document["text"]:
        from chunking import chunk_document

        st.subheader("Prepare chunks")
        st.write("Each chunk keeps a small overlap with the previous one to preserve nearby context.")
        with st.form("chunk_" + identifier):
            size = st.number_input("Words per chunk", min_value=20, max_value=1000, value=150, step=10)
            overlap = st.number_input("Overlapping words", min_value=0, max_value=500, value=30, step=5)
            processed = st.form_submit_button("Create chunks", type="primary")
        if processed:
            def process():
                chunks = chunk_document(document, size, overlap)
                storage.write_document(
                    {**document, "chunks": chunks, "chunk_size": size, "overlap": overlap},
                    document,
                )
            mutate(process, "Chunks saved.")
        st.caption(f"{len(document['chunks'])} chunks saved")
        for chunk in document["chunks"]:
            with st.expander(f"Chunk {chunk['number']} · words {chunk['start_word'] + 1}–{chunk['end_word']}"):
                labels = [f"Source {identifier}"]
                if chunk.get("sheet"):
                    labels.append(f"Sheet {chunk['sheet']}")
                elif chunk.get("page") is not None:
                    labels.append(f"Page {chunk['page']}")
                labels.append(f"Chunk {chunk['number']}")
                st.caption(" · ".join(labels))
                st.text(chunk["text"])


def mutate(action, message):
    try:
        action()
    except Exception as error:
        # Parsers raise different errors for damaged or encrypted files.
        st.error(f"Could not complete this action. {error}")
    else:
        st.session_state.document_notice = message
        st.rerun()
