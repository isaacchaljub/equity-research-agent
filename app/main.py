"""Streamlit chat UI for the equity-research analyst (talks to serving_api/main.py).

Carries the conversation_id in st.session_state, so follow-ups keep context without the caller
passing ids around. Closing/reopening the tab resets session_state and starts a fresh
conversation. Each answer's fact-check verdict is shown beneath it.

Run the API first, then this UI:
    fastapi run serving_api/main.py --host 0.0.0.0 --port 8000
    streamlit run app/main.py
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Equity Research Agent", page_icon="📈")
st.title("📈 Equity Research Agent")
st.caption("RAG over filings + live market data + web news, with a fact-checking verifier. Educational, not financial advice.")

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.caption(f"API: {API_URL}")
    if st.session_state.conversation_id:
        st.caption(f"Conversation: {st.session_state.conversation_id[:8]}…")
    if st.button("New conversation"):
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("verification"):
            with st.expander("Fact-check"):
                st.markdown(message["verification"])

prompt = st.chat_input("Ask about a company, ticker, or filing…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"), st.spinner("Researching and verifying…"):
        payload = {"message": prompt}
        if st.session_state.conversation_id:
            payload["conversation_id"] = st.session_state.conversation_id
        verification = ""
        try:
            response = requests.post(f"{API_URL}/chat", json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            st.session_state.conversation_id = data["conversation_id"]
            answer = data["answer"]
            verification = data.get("verification", "")
        except Exception as e:
            answer = f"Error talking to the API at {API_URL}: {e}"
        st.markdown(answer)
        if verification:
            with st.expander("Fact-check"):
                st.markdown(verification)

    st.session_state.messages.append({"role": "assistant", "content": answer, "verification": verification})
