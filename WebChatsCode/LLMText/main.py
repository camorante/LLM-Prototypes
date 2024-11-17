from langchain_openai import ChatOpenAI
import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

database_url = "postgresql+psycopg2://llmuser:123456789@localhost:5432/dvdrental"
db = SQLDatabase.from_uri(database_url)
res = db.run("SELECT * FROM public.actor ORDER BY actor_id ASC limit 1")
print(res)

key = open("key.txt", "r")
os.environ["OPENAI_API_KEY"] = key.read()
key.close()

key = open("keyls.txt", "r")
os.environ["LANGCHAIN_PROJECT"] = "gpstrackit-dev"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = key.read()
key.close()

llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0.2)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
print(tools)

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

agent_executor = create_react_agent(llm, tools, messages_modifier=system_message)

for s in agent_executor.stream(
    {"messages": [HumanMessage(content='cuales peliculas tiene rentandas el usuario con correo "mary.smith@sakilacustomer.org"')]}
):
    print(s)
    print("----")