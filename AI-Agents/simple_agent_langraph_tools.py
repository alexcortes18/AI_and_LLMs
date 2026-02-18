import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph.message import add_messages
import pprint

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
llm_name = "gpt-3.5-turbo"
client = OpenAI(api_key=openai_key)
model = ChatOpenAI(model= llm_name, api_key=openai_key)

# Step 1: Build a Classic Chatbot
class State(TypedDict):
    messages: Annotated[list, add_messages]
    
# Create tools
tool = TavilySearchResults(max_results = 2)
tools = [tool] # We use a list because usually a model can take several tools
# res = tool.invoke("What is the capital of France?") # Testing that Tavily works
# pprint.pprint(res)

model_with_tools = model.bind_tools(tools) #add_tools seems deprecated
res = model_with_tools.invoke("What is a 'node'in Langgraph?") # Testing that Tavily works
pprint.pprint(res)
    
def bot(state: State):
    print(state["messages"])
    return {"messages": [model.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("bot", bot) # Step 2
graph_builder.set_entry_point("bot") # Step 3
graph_builder.set_finish_point("bot") # Step 4
graph = graph_builder.compile() # Step 5

