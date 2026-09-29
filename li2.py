import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load Environment Variables from .env
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("OPENROUTER_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

# 2. Initialize the OpenAI Client configured for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Configure Streamlit Page Title & Icon
st.set_page_config(page_title="OpenRouter Streaming Chat", page_icon="💬")
st.title("💬 OpenRouter Streaming Chat")

# 3. Initialize Persistent Conversation History in Streamlit Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 4. Display Existing Chat Messages from History
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Handle New User Input
if user_prompt := st.chat_input("Ask anything..."):
    # Display user's message immediately in the UI
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    # Store user message in history
    st.session_state["messages"].append({"role": "user", "content": user_prompt})

    # Prepare formatted message payload
    formatted_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state["messages"]
    ]

    # 6. Stream Response from OpenRouter
    with st.chat_message("assistant"):
        def response_generator():
            response_stream = client.chat.completions.create(
                model="google/gemini-2.5-flash",
                messages=formatted_messages,
                stream=True,
                max_tokens=1000  # Restricts token count to prevent OpenRouter 402 error
            )
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        # Stream tokens directly onto screen UI
        full_response = st.write_stream(response_generator())

    # Store assistant response in history
    st.session_state["messages"].append({"role": "assistant", "content": full_response})