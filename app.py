import streamlit as st
import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage

# ========== FILE TO STORE CHAT HISTORY ==========
CHAT_HISTORY_FILE = "chat_history.json"

# ========== LOAD CHAT HISTORY ==========
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    return []

# ========== SAVE CHAT HISTORY ==========
def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(history, file)

# ========== STREAMLIT PAGE ==========
st.set_page_config(page_title="My Notes AI", page_icon="🤖")
st.title("My Notes AI 🤖")

# Load history from file
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key="YOUR_GOOGLE_API_KEY"
)

# User input
if prompt := st.chat_input("Ask something..."):
    
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Convert history to LangChain format
    chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    # Get AI response
    response = llm(chat_history)

    # Show AI message
    st.chat_message("assistant").markdown(response.content)
    st.session_state.messages.append({"role": "assistant", "content": response.content})

    # Save to file
    save_chat_history(st.session_state.messages)

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    save_chat_history([])
    st.rerun()
