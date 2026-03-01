import streamlit as st
import os
groq_key = st.secrets["GROQ_API_KEY"]

from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


st.set_page_config(page_title="My Notes AI", layout="wide")

st.title("📚 My Personal Notes AI Assistant")

# -------- LOAD EMBEDDINGS --------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -------- LOAD VECTOR DB --------
db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

# -------- LOAD LLM (Groq) --------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# -------- SESSION STATE --------
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None


# -------- SIDEBAR --------
st.sidebar.title("💬 Chat History")

# New Chat Button
if st.sidebar.button("➕ New Chat"):
    st.session_state.current_chat = None

# Show Previous Chats
for chat_title in st.session_state.conversations.keys():
    if st.sidebar.button(chat_title):
        st.session_state.current_chat = chat_title


# -------- MAIN CHAT INPUT --------
user_input = st.text_input("Ask something from your notes:")

if user_input:

    # If first message → create new chat
    if st.session_state.current_chat is None:
        chat_title = user_input[:30]
        st.session_state.conversations[chat_title] = []
        st.session_state.current_chat = chat_title

    chat = st.session_state.conversations[st.session_state.current_chat]

    chat.append(("You", user_input))

    # Retrieve relevant docs
    docs = db.similarity_search(user_input, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Build conversation history
    history_text = ""
    for role, message in chat:
        history_text += f"{role}: {message}\n"

    prompt = f"""
You are a helpful assistant. Use ONLY the provided context to answer.

Context:
{context}

Conversation:
{history_text}

Answer clearly and concisely:
"""

    response = llm.invoke(prompt).content

    chat.append(("Assistant", response))


# -------- DISPLAY CHAT --------
if st.session_state.current_chat:
    chat = st.session_state.conversations[st.session_state.current_chat]
    for role, message in chat:
        if role == "You":
            st.markdown(f"**🧑 {role}:** {message}")
        else:
            st.markdown(f"**🤖 {role}:** {message}")