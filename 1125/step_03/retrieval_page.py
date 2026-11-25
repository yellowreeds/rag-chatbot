"""Index controls, followed by semantic search in checkpoint 04."""

import streamlit as st

from indexing import EMBED_MODEL, build_index, load_index


def retrieval_page(stage):
    st.title("Explore your document index")
    st.write(f"Embeddings use {EMBED_MODEL}. Build again after uploading, deleting, extracting, or recreating chunks.")
    if st.button("Build index", type="primary"):
        try:
            with st.spinner("Embedding document chunks…"):
                build_index()
        except Exception as error:
            st.error(f"Index build failed. {error}")
        else:
            st.success("Index saved.")
    try:
        index, _ = load_index()
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.info(str(error))
        return
    st.caption(f"{len(index['records'])} indexed chunks · {index['dimensions']} dimensions · {index['model']}")
    if stage >= 4:
        from search import search, show_results
        with st.form("search"):
            query = st.text_input("Question in English or Korean")
            submitted = st.form_submit_button("Search passages")
        if submitted:
            try:
                with st.spinner("Finding relevant passages…"):
                    results = search(query)
            except Exception as error:
                st.error(f"Search failed. {error}")
            else:
                if not results:
                    st.info("No passages met the similarity threshold. Try another question.")
                show_results(results)
