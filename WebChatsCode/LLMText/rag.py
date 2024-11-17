import bs4
import chromadb
import streamlit as st
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory

prompt = hub.pull("rlm/rag-prompt")
chromadb.api.client.SharedSystemClient.clear_system_cache()
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def createAgent():
    llm = ChatOllama(model="hf.co/SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF:Q6_K", temperature=0.8)
    embeddings = OllamaEmbeddings( model="hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:Q6_K")
    web_loader = WebBaseLoader(
        web_paths=("https://lavidaesunvideojuego.com/2024/11/07/la-gran-oscuridad-ha-llegado-a-hearthstone/",
                   "https://lavidaesunvideojuego.com/2024/09/05/tips-para-tu-primer-ano-en-stardew-valley/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    web_documents = web_loader.load()

    text_loader_kwargs = {'autodetect_encoding': True}
    text_loader = DirectoryLoader("./", glob="docs/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    text_documents = text_loader.load()

    all_documents = web_documents + text_documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(all_documents)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm.bind(stop=["Answer:", "<|im_end|>"])
            | StrOutputParser()
    )
    return rag_chain


def main():
    st.set_page_config(page_title="RAG Chat")
    st.title("💬 RAG Chatbot")
    st.caption("🚀 Chatbot Using Local LLM and RAG")
    chat_history = ChatMessageHistory()
    llm = createAgent()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        #st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        full_out = st.chat_message("assistant").write_stream(llm.stream(prompt))
        print(full_out)
        st.session_state.messages.append({"role": "assistant", "content": full_out})

if __name__  == '__main__':
    main()