import streamlit as st
import requests
from typing import List, Dict
import json

# Configure the page
st.set_page_config(
    page_title="Economics Expert Chat",
    page_icon="📚",
    layout="centered"
)

# Add title and description
st.title("Economics Expert Chat 📚")
st.markdown("""
Ask questions about economics concepts and get answers based on the textbook 'Principles of Economics'.
""")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input
if prompt := st.chat_input("Ask your economics question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send request to FastAPI backend
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={"message": prompt},
            timeout=30
        )
        response.raise_for_status()

        # Parse response
        result = response.json()
        answer = result["response"]
        sources = result["sources"]

        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
    except Exception as e:
        st.error(f"Error: Failed to get response from the server. {str(e)}")

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])
            if "sources" in message:
                st.markdown("---")
                st.markdown("**Sources:**")
                for source in message["sources"]:
                    st.markdown(f"- {source}")