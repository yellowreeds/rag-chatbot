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
        file = st.file_uploader("Add a document", type=["txt", "pdf", "docx", "hwp", "hwpx"])
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
        st.info("Your library is ready for its first document. Upload a TXT, PDF, DOCX, HWP, or HWPX file.")
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
        from processing import extract_text

        if st.button("Extract text", type="primary"):
            def extract():
                with st.spinner("Reading the document…"):
                    text = extract_text(document["original_name"], base64.b64decode(document["content"]))
                storage.write_document(
                    {**document, "text": text, "chunks": [], "chunk_size": None, "overlap": None},
                    document,
                )
            mutate(extract, "Text extracted. Create chunks from this version of the text.")
        if document["text"]:
            st.caption(f"{len(document['text'].split()):,} words extracted")
            with st.expander("Extracted text", expanded=True):
                st.text(document["text"])



def mutate(action, message):
    try:
        action()
    except Exception as error:
        # Parsers raise different errors for damaged or encrypted files.
        st.error(f"Could not complete this action. {error}")
    else:
        st.session_state.document_notice = message
        st.rerun()
