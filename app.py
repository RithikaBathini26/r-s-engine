import streamlit as st
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="My Notes AI", layout="wide")
st.title("📚 My Personal Notes AI Assistant")

# ------------------ LOAD SECRET KEY ------------------
groq_key = st.secrets["GROQ_API_KEY"]

# ------------------ IMPORTS ------------------
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ------------------ CACHE HEAVY COMPONENTS ------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_vector_db():
    return FAISS.load_local(
        "vector_db",
        load_embeddings(),
        allow_dangerous_deserialization=True
    )

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=groq_key
    )

embeddings = load_embeddings()
db = load_vector_db()
llm = load_llm()

# ------------------ SESSION STATE ------------------

if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

# ------------------ SIDEBAR ------------------

st.sidebar.title("💬 Chat History")

# New Chat
if st.sidebar.button("➕ New Chat"):
    st.session_state.current_chat = None

# Clear Current Chat
if st.sidebar.button("🗑 Clear Current Chat"):
    if st.session_state.current_chat:
        st.session_state.conversations[st.session_state.current_chat] = []
        st.rerun()

# Show Previous Chats
for chat_title in st.session_state.conversations.keys():
    if st.sidebar.button(chat_title):
        st.session_state.current_chat = chat_title

# ------------------ CHAT INPUT ------------------

user_input = st.chat_input("Ask something from your notes...")

if user_input:

    # Create new chat if first message
    if st.session_state.current_chat is None:
        chat_title = user_input[:30]
        st.session_state.conversations[chat_title] = []
        st.session_state.current_chat = chat_title

    chat = st.session_state.conversations[st.session_state.current_chat]

    # Add user message
    chat.append(("You", user_input))

    # ------------------ RETRIEVE CONTEXT ------------------
    docs = db.similarity_search(user_input, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # ------------------ BUILD HISTORY ------------------
    history_text = ""
    for role, message in chat:
        history_text += f"{role}: {message}\n"

    # ------------------ PROMPT ------------------
    prompt = f"""
You are a helpful AI assistant.

Use ONLY the provided context to answer.
If the answer is not in the context, say:
"I could not find this information in your notes."

---------------------
CONTEXT:
{context}
---------------------

CONVERSATION HISTORY:
{history_text}

Now answer clearly and concisely:
"""

    # ------------------ LLM RESPONSE ------------------
    response = llm.invoke(prompt).content

    chat.append(("Assistant", response))

# ------------------ DISPLAY CHAT ------------------

if st.session_state.current_chat:
    chat = st.session_state.conversations[st.session_state.current_chat]

    for role, message in chat:
        if role == "You":
            with st.chat_message("user"):
                st.markdown(message)
        else:
            with st.chat_message("assistant"):
                st.markdown(message)
