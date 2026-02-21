import os
import json
import operator
from io import StringIO
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient

memory = InMemorySaver()
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
tavily = os.getenv("TAVILY_API_KEY")
llm_name = "gpt-3.5-turbo"

model = ChatOpenAI(api_key= openai_key, model= llm_name)
tavily = TavilyClient(api_key=tavily)

#schema for the whole graph state dict passed between nodes
class AgentState(TypedDict):
    task: str
    competitors: List[str]
    csv_file: str
    financial_data: str
    analysis: str
    competitor_data: str
    comparison: str
    feedback: str
    report: str
    content: List[str]
    revision_number: int
    max_revisions: int

# schema for one structured payload (e.g., LLM output)
class Queries(BaseModel):
    queries: List[str]
    
# Define the prompts for each node - IMPROVE AS NEEDED
GATHER_FINANCIALS_PROMPT = """You are an expert financial analyst. Gather the financial data for the given company. Provide detailed financial data."""
ANALYZE_DATA_PROMPT = """You are an expert financial analyst. Analyze the provided financial data and provide detailed insights and analysis."""
RESEARCH_COMPETITORS_PROMPT = """You are a researcher tasked with providing information about similar companies for performance comparison. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""
COMPETE_PERFORMANCE_PROMPT = """You are an expert financial analyst. Compare the financial performance of the given company with its competitors based on the provided data.
**MAKE SURE TO INCLUDE THE NAMES OF THE COMPETITORS IN THE COMPARISON.**"""
FEEDBACK_PROMPT = """You are a reviewer. Provide detailed feedback and critique for the provided financial comparison report. Include any additional information or revisions needed."""
WRITE_REPORT_PROMPT = """You are a financial report writer. Write a comprehensive financial report based on the analysis, competitor research, comparison, and feedback provided."""
RESEARCH_CRITIQUE_PROMPT = """You are a researcher tasked with providing information to address the provided critique. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""

def gather_financials_node(state: AgentState):
    csv_file = state["csv_file"]
    df = pd.read_csv(StringIO(csv_file))
    financial_data_str = df.to_string(index=False)
    
    combined_content = (
        f"""{state['task']} \n\n Here is the financial data:
        \n\n{financial_data_str}"""
    )
    
    messages = [
        SystemMessage(content= GATHER_FINANCIALS_PROMPT),
        HumanMessage(content=combined_content)
    ]
    
    response = model.invoke(messages)
    return {"financial_data" : response.content} # a key of AgentState

def analyze_data_node(state: AgentState):
    messages = [
        SystemMessage(content=ANALYZE_DATA_PROMPT),
        HumanMessage(content=state["financial_data"])
    ]
    response = model.invoke(messages)
    return {"analysis": response.content} # a key of AgentState
    
def research_competitors_node(state: AgentState):
    content = state["content"] or []
    for competitor in state["competitors"]:
        queries = model.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=RESEARCH_COMPETITORS_PROMPT),
                HumanMessage(content=competitor),
            ]
        )
        for q in queries.queries:
            response = tavily.search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])
    return {"content": content}

def compare_performance_node(state: AgentState):
    content = "\n\n".join(state["content"] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is the financial analysis:\n\n{state['analysis']}"
    )
    messages = [
        SystemMessage(content=COMPETE_PERFORMANCE_PROMPT.format(content=content)),
        user_message,
    ]
    response = model.invoke(messages)
    return {
        "comparison": response.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }





# builder = StateGraph(AgentState)
# builder.add_node("gather_financials", gather_financials_node)