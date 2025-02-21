import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"  # Update with your FastAPI endpoint

def get_chat_response(user_input):
    response = requests.post(API_URL, json={"message": user_input})
    if response.status_code == 200:
        return response.json().get("response", "Error: No response from server")
    return "Error: Unable to fetch response"

st.title("PuntPunt Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    response = get_chat_response(user_input)
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.rerun()


# Run with: streamlit run chat-ui.py