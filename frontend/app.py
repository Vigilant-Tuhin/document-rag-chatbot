import asyncio

import streamlit as st

from utils.api_client import delete_session, get_history, list_sessions, upload_file
from utils.websocket_client import stream_chat

### SESSION STATE INITIALISATION
if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "refresh_sessions" not in st.session_state:
    st.session_state.refresh_sessions = True

if "reset_uploader" not in st.session_state:
    st.session_state.reset_uploader = False

if "sessions_cache" not in st.session_state:
    st.session_state.sessions_cache = {"sessions": []}


# Hide uploader after upload completes (forces a fresh widget instance next time)
if st.session_state.reset_uploader:
    st.session_state.pop("chat_uploader", None)
    st.session_state.reset_uploader = False


@st.cache_data
def cached_list_sessions():
    return list_sessions()


if st.session_state.refresh_sessions:
    st.session_state.sessions_cache = cached_list_sessions()
    st.session_state.refresh_sessions = False

session_items = st.session_state.sessions_cache["sessions"]


### HEADER
st.title("📄 Document RAG Chatbot")

if st.session_state.session_id:
    name = next((s["name"] for s in session_items if s["id"] == st.session_state.session_id), None)
    st.write(f"#### 🔥 Active chat: {name or st.session_state.session_id}")


### MAIN AREA — upload (only when no session is active; one document per session)
allow_upload = st.session_state.session_id is None

if allow_upload:
    uploaded = st.file_uploader("📤 Upload your document", key="chat_uploader")

    if uploaded:
        try:
            result = upload_file(uploaded)
        except RuntimeError as e:
            st.error(f"Couldn't process that file: {e}")
        else:
            st.session_state.session_id = result["session_id"]
            st.session_state.chat_history = []
            cached_list_sessions.clear()
            st.session_state.refresh_sessions = True
            st.session_state.reset_uploader = True
            st.success("Document ingested! Start chatting.")
            st.rerun()


### SIDEBAR
with st.sidebar:

    if st.button("New Chat", icon=":material/add_2:", width="stretch"):
        st.session_state.session_id = None
        st.session_state.chat_history = []
        st.session_state.reset_uploader = True
        st.rerun()

    if st.session_state.session_id and st.button("Delete This Chat", width="stretch", type="primary"):
        delete_session(st.session_state.session_id)

        st.session_state.session_id = None
        st.session_state.chat_history = []
        cached_list_sessions.clear()
        st.session_state.refresh_sessions = True
        st.session_state.reset_uploader = True
        st.rerun()

    st.subheader("💬 Your Chats")

    for item in session_items:
        sid, name = item["id"], item["name"]
        if st.button(name, key=f"session_{sid}", width="stretch"):
            st.session_state.session_id = sid
            st.session_state.chat_history = get_history(sid)["history"]
            st.session_state.reset_uploader = True
            st.rerun()


### CHAT HISTORY
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


### CHAT INPUT
user_msg = st.chat_input("Ask something...")

if user_msg:
    if not st.session_state.session_id:
        st.error("Please upload a document first.")
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        buffer = [""]

        async def _stream():
            async for token in stream_chat(st.session_state.session_id, user_msg):
                buffer[0] += token
                placeholder.write(buffer[0])

        asyncio.run(_stream())
        st.session_state.chat_history.append({"role": "assistant", "content": buffer[0]})

    st.rerun()
