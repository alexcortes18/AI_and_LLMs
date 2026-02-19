import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
import pprint
import json

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
# pprint.pprint(res)

def bot(state: State):
    print(state["messages"])
    return {"messages": [model_with_tools.invoke(state["messages"])]}

graph_builder = StateGraph(State)

# Not needed since we can just use the prebuilt from langchain
# class BasicToolNode:
#     """A node that runs the tools requested in the last AIMessage."""
#     def __init__(self, tools: list) -> None:
#         self.tools_by_name = {tool.name: tool for tool in tools}
#     def __call__(self, inputs: dict):
#         if messages := inputs.get("messages", []):
#             message = messages[-1]
#         else:
#             raise ValueError("No message found in input")
#         outputs = []
#         for tool_call in message.tool_calls:
#             tool_result = self.tools_by_name[
#                 tool_call["name"]
#             ].invoke(tool_call["args"])
#             outputs.append(
#                 ToolMessage(
#                     content=json.dumps(tool_result),
#                     name=tool_call["name"],
#                     tool_call_id=tool_call["id"],
#                 )
#             )
#         return {"messages": outputs}

# Instantiate the BasicToolNode with the tools
# tool_node = BasicToolNode(tools=[tool]) # Before we used this
tool_node = ToolNode(tools=[tool]) # instead of BasicToolNode
graph_builder.add_node("tools",tool_node)

# Not needed since we can just use the prebuilt from langchain
# def route_tools(
#     state: State,
# ) -> Literal["tools", "__end__"]:
#     """
#     Use in the conditional_edge to route to the ToolNode if the last message
#     has tool calls. Otherwise, route to the end.
#     """
#     if isinstance(state, list):
#         ai_message = state[-1]
#     elif messages := state.get("messages", []):
#         ai_message = messages[-1]
#     else:
#         raise ValueError(
#             f"No messages found in input state to tool_edge: {state}"
#         )
#     if (
#         hasattr(ai_message, "tool_calls")
#         and len(ai_message.tool_calls) > 0
#     ):
#         return "tools"
#     return "__end__"

graph_builder.add_conditional_edges(
    "bot",
    tools_condition #now we use this instead of route_tools
    # These two before was with useing route_tools
    # route_tools,
    # {"tools":"tools", "__end__":"__end__"}
)

graph_builder.add_node("bot", bot)
graph_builder.set_entry_point("bot")
# graph_builder.set_finish_point("bot") # Not needed anymore, because the tools should know how to end
graph = graph_builder.compile()

from langchain_core.messages import BaseMessage

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            if isinstance(value["messages"][-1], BaseMessage):
                pprint.pprint(f"Assistant: {value['messages'][-1].content}")