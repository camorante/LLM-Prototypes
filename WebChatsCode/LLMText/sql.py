import os
import time

import streamlit as st
from langchain_ollama import ChatOllama
from langchain.llms import Ollama
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

key = open("key.txt", "r")
os.environ["OPENAI_API_KEY"] = key.read()
key.close()

def createAgent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
    database_url = "postgresql+psycopg2://llmuser:123456789@localhost:5432/dvdrental"
    db = SQLDatabase.from_uri(database_url)
    db.run("SELECT * FROM public.actor ORDER BY actor_id ASC limit 1")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    SQL_PREFIX = """You are an agent designed to interact with a PostgreSQL database.
    Given an input question, create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
    
    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
    
    To start you should ALWAYS look at the tables in the database to see what you can query.
    Do NOT skip this step.
    Then you should query the schema of the most relevant tables."""

    system_message = SystemMessage(content=SQL_PREFIX)
    agent_executor = create_react_agent(llm, tools, state_modifier=system_message)
    return agent_executor

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

def main():
    st.set_page_config(page_title="Basic Chat")
    st.title("💬 Basic Chatbot ")
    st.caption("🚀 Chatbot Using Local LLM")
    llm = createAgent()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        #st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        last_resp = ""
        for s in llm.stream(
                {"messages": [HumanMessage(content=prompt)]}
        ):
            last_resp = s
        if "agent" in last_resp:
            full_out = last_resp['agent']['messages'][0].content

        st.chat_message("assistant").write_stream(stream_data(full_out))
        print(full_out)
        st.session_state.messages.append({"role": "assistant", "content": full_out})

if __name__  == '__main__':
    main()