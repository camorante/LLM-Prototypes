import streamlit as st
from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory

def main():
    st.set_page_config(page_title="Basic Chat")
    st.title("💬 Basic Chatbot ")
    st.caption("🚀 Chatbot Using Local LLM")
    chat_history = ChatMessageHistory()
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        llm = ChatOllama(model="hf.co/SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF:Q6_K", temperature=0.8)
        chat_history.add_user_message(prompt)
        #llm.invoke(chat_history.messages)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        full_out = st.chat_message("assistant").write_stream(llm.stream(st.session_state.messages))
        print(full_out)
        chat_history.add_ai_message(full_out)
        st.session_state.messages.append({"role": "assistant", "content": full_out})

if __name__  == '__main__':
    main()