from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

load_dotenv()

model =  ChatGroq(model = 'llama-3.1-8b-instant',
    temperature=0,
    max_retries=2)
# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Callback to send message and clear input
def send_message():

    user_input = st.session_state.user_input
    if user_input:
        # Include chat history in the input
        history_text = ""
        for msg in st.session_state.messages:
            role = "You" if msg["role"] == "user" else "AI"
            history_text += f"{role}: {msg['content']}\n"


        prompt = history_text + f"You: {user_input}\nAI:"

        # Get AI response using combined context
        result = model.invoke(prompt)

        # Save conversation
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "bot", "content": result.content})

        # Clear input box
        st.session_state.user_input = ""
# Input box with callback
st.text_input("You:", key="user_input", placeholder="Type your message here...", on_change=send_message)

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**AI:** {msg['content']}")

