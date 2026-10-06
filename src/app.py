import streamlit as st
from ingest import chunking_uploaded_pdf, create_embeddings
from rag import retrieve, build_context, build_prompt, generate_answer
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

st.title("Multi-paper Research Assistant")
uploaded_files = st.file_uploader(
    "Upload research papers (PDF)",
    type = "pdf",
    accept_multiple_files = True
)

if uploaded_files:
    current_files = [file.name for file in uploaded_files]
    model = load_embedding_model()

    if "processed_files" not in st.session_state or current_files != st.session_state["processed_files"]:
        with st.spinner("Processing papers..."):
            all_chunks = []
            for file in uploaded_files:
                chunk_list   = chunking_uploaded_pdf(file, file.name)
                if not chunk_list:
                    st.warning(
                        f"Could not extract readable text from {file.name}. "
                        "Please upload a text-based/searchable PDF."
                    )
                    continue
                all_chunks.extend(chunk_list)
            if all_chunks:
                st.session_state["all_chunks"] = all_chunks
                st.session_state["processed_files"] = current_files
                embeddings = create_embeddings(st.session_state["all_chunks"], model)
                st.session_state["embeddings"] = embeddings
                
            else:
                st.session_state.pop("all_chunks", None)
                st.session_state.pop("embeddings", None)
    if "all_chunks" in st.session_state:
        st.success(
            f"{len(current_files)} paper(s) ready — "
            f"{len(st.session_state['all_chunks'])} chunks indexed"
        )
            
        query = st.text_input("Ask a question about your papers")

        ask_button = st.button("Ask")

        if ask_button and query:
            top_indices = retrieve(query, model, st.session_state["embeddings"])
            context = build_context(top_indices,st.session_state["all_chunks"])
            
            prompt = build_prompt(context, query)
            
            llm_model = "nvidia/nemotron-3-super-120b-a12b:free"
            #llm_mode = "openrouter/free" #if nividia stops functioning
            
            load_dotenv()
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                api_key = st.secrets["OPENROUTER_API_KEY"]
            
            status_code, model_name, answer = generate_answer(prompt, llm_model, api_key)
            if status_code == 200:
                st.subheader("Answer:")
                st.markdown(answer)

            else:
                st.error("Failed to generate an answer.")
    else:
        st.error("No readable text could be extracted from the uploaded PDFs.")
    