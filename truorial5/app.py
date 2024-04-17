import streamlit as st
from openai import OpenAI

with st.sidebar:
    model_name = st.sidebar.selectbox(
        '选择一个选项',
        ('internlm2', 'yykx', 'hdnj')
    )

st.title("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    #client = OpenAI(api_key='internlm2-chat-1_8b')
    client = OpenAI(
        api_key='YOUR_API_KEY',
        base_url="http://0.0.0.0:23333/v1"
    )
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(
        model=model_name,
        messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)