import os
import pprint
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langgraph.graph import StateGraph, END

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

llm_model = "gpt-4o-mini"
# client = OpenAI(api_key=openai_key) # Not being used
model = ChatOpenAI(api_key=openai_key, model=llm_model)

# Step 1: Build a Basic Chaptbot
from langgraph.graph.message import add_messages

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    
def bot(state: State):
    print(state["messages"])
    return {"messages":[model.invoke(state["messages"])]}

graph_builder = StateGraph(State)

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("bot", bot)

# Step 3: Add an entry point to the graph
graph_builder.set_entry_point("bot")

# Step 4: Add end point to the graph
graph_builder.set_finish_point("bot")
    
# Step 5: Compile the graph
graph = graph_builder.compile()
print(type(graph), "\n")

# response = graph.invoke({"messages": ["Hello, how are you?"]})
# print("\n")
# print(response["messages"])

while True:
    user_input  = input("User:")
    print("\n")
    if user_input.lower() in ["quit","exit", "q"]:
        pprint.pprint("You have exited!")
        break
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            # pprint.pprint(f"Assistant: {value['messages'][-1].content}")
            print(f"Assistant: {value['messages'][-1].content}")