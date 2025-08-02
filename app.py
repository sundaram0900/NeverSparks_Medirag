import streamlit as st
from rag_pipeline import build_qa_chain

st.set_page_config(page_title="MediRAG: Medical Literature Assistant", layout="wide")
st.title("🩺 MediRAG")
st.markdown("Ask evidence-based questions from loaded medical literature.")

query = st.text_input("🔎 Enter your clinical query:", placeholder="e.g., What are the symptoms of COVID-19?")

if query:
    qa_chain = build_qa_chain()
    with st.spinner("Thinking..."):
        result = qa_chain(query)
        st.success("✅ Answer:")
        st.write(result['result'])

        with st.expander("📄 Sources"):
            for doc in result['source_documents']:
                st.markdown(f"**Source:** {doc.metadata.get('source', 'N/A')}")
                st.write(doc.page_content)
