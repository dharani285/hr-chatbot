"""
ChatGPT-like Streamlit UI for the HR RAG Chatbot.

Run:
    streamlit run streamlit.py --server.port 8501

Backend:
    uvicorn main:app --reload --port 8000

Notes:
- Set GEMINI_API_KEY in .env or environment for backend answers.
- API base URL can be changed from the sidebar (env var RAG_API_URL is default).
"""

import os
import requests
import streamlit as st

# --- Page setup ---
st.set_page_config(page_title="HR RAG Chatbot", page_icon="🤖", layout="centered")

API_URL_DEFAULT = os.environ.get("RAG_API_URL", "http://localhost:8000")

# --- Sidebar settings ---
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API Base URL", value=API_URL_DEFAULT)
    k = st.slider("Top-K Chunks", min_value=1, max_value=20, value=5)
    show_context = st.checkbox("Attach Context Under Answer", value=False)
    show_sources = st.checkbox("Attach Sources Under Answer", value=True)

    col_a, col_b = st.columns(2)
    with col_a:
        health_clicked = st.button("Check API Health")
    with col_b:
        clear_clicked = st.button("Clear Chat")

    if health_clicked:
        try:
            resp = requests.get(f"{api_url}/health", timeout=10)
            if resp.ok:
                data = resp.json()
                st.success(
                    f"API OK | Index: {data.get('has_index')} | Meta: {data.get('has_metadata')} | Emb: {data.get('has_embeddings')}"
                )
            else:
                st.error(f"Health check failed: {resp.status_code} {resp.text}")
        except Exception as e:
            st.error(f"Health check error: {e}")

# --- Session state for chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your HR RAG assistant. Ask me about your HR documents."}
    ]

if clear_clicked:
    st.session_state.messages = [
        {"role": "assistant", "content": "Chat cleared. How can I help you now?"}
    ]

st.title("🤖 HR RAG Chatbot")
st.caption("Chat with your HR knowledge base.")

# --- Render chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])  # main content
        # Optionally render context and sources under assistant messages
        if msg.get("role") == "assistant":
            if msg.get("context"):
                with st.expander("Context used"):
                    st.code(msg["context"], language="markdown")
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        score = s.get("score", 0.0)
                        doc_id = s.get("doc_id", "")
                        chunk_id = s.get("chunk_id", "")
                        st.markdown(f"- Score: `{score:.3f}` | Source: `{doc_id}` | Chunk: `{chunk_id}`")

# --- Chat input ---
user_input = st.chat_input("Type your question...")

if user_input:
    # 1) Echo the user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Call backend
    payload = {
        "question": user_input.strip(),
        "k": k,
        "return_context": True,  # always fetch; toggle only affects display
        "return_sources": True,
    }

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = requests.post(f"{api_url}/ask", json=payload, timeout=60)
            if not resp.ok:
                st.error(f"Request failed: {resp.status_code} {resp.text}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Sorry, I ran into an error: {resp.status_code}",
                })
            else:
                data = resp.json()
                answer = data.get("answer", "(No response)")
                rendered = st.markdown(answer)

                # Attach optional panels below the assistant bubble
                used_context = data.get("context") if show_context else None
                used_sources = data.get("sources") if show_sources else None

                if used_context:
                    with st.expander("Context used"):
                        st.code(used_context, language="markdown")
                if used_sources:
                    with st.expander("Sources"):
                        for s in used_sources:
                            score = s.get("score", 0.0)
                            doc_id = s.get("doc_id", "")
                            chunk_id = s.get("chunk_id", "")
                            st.markdown(f"- Score: `{score:.3f}` | Source: `{doc_id}` | Chunk: `{chunk_id}`")

                # 3) Persist assistant message with optional attachments
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "context": data.get("context"),
                    "sources": data.get("sources"),
                })
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.caption("Backend: FastAPI | Frontend: Streamlit · Chat mode")