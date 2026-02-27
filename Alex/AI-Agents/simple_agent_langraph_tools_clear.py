import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import pprint

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
llm_name = "gpt-3.5-turbo"

model = ChatOpenAI(model=llm_name, api_key=openai_key)


class State(TypedDict):
    messages: Annotated[list, add_messages]


tool = TavilySearchResults(max_results=2)
tools = [tool]
model_with_tools = model.bind_tools(tools) # -> “These tools exist; here are their names and arg schemas.”


def bot(state: State):
    print(state["messages"])
    return {"messages": [model_with_tools.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("bot", bot)
graph_builder.add_node("tools", ToolNode(tools=tools))   # if rounting or control flow arrives here, then this node executes requested tool_calls and returns ToolMessage(s).
graph_builder.set_entry_point("bot")

# "tools_condition" checks the last AI message from "bot":
# - if tool_calls exist -> route to "tools"
# - else -> route to "__end__"
# If tools_condition sees tool_calls, that means the LLM already decided to use a tool
# and is influenced by tool schemas (bind_tools), prompts/instructions, and model settings.
graph_builder.add_conditional_edges("bot", tools_condition)
graph_builder.add_edge("tools", "bot")
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    
    # Important flow/routing block:
    # Streams one graph run from an initial user message in state, yielding node-by-node events.
    for event in graph.stream(
        {"messages": [("user", user_input)]},
        config=config,
    ): # user_input enters state, then entry node ("bot") runs
        for value in event.values():
            if isinstance(value["messages"][-1], BaseMessage):
                pprint.pprint(f"Assistant: {value['messages'][-1].content}")
