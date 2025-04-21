from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END

# We will use this model for both the conversation and the summarization
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# from langchain_ollama import OllamaLLM
from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode

from langchain_community.tools import DuckDuckGoSearchRun

from langchain_mcp_adapters.client import MultiServerMCPClient

from contextlib import asynccontextmanager


import asyncio


@tool
def search_over_internet(search_query: str):
    """
    search over internet for the given query
    Args:
        search_query: query to search over internet
    """
    search = DuckDuckGoSearchRun()
    return search.invoke(search_query)


model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


@asynccontextmanager
async def main():
    async with MultiServerMCPClient(
        {
            "playwright": {"command": "npx", "args": ["@playwright/mcp@latest"]},
            # "sequential-thinking": {
            #   "command": "npx",
            #    "args": [
            #        "-y",
            #        "@modelcontextprotocol/server-sequential-thinking",
            #        "--persist",
            #    ],
            # },
            # "weather": {
            #    "command": "uv",
            #    "args": [
            #        "--directory",
            #        "/home/rahul/agents/weather",
            #        "run",
            #        "weather.py",
            #    ],
            # "url": "http://localhost:8000/sse",
            # "transport": "sse",
            # },
        }
    ) as client:

        tools = client.get_tools()  # [search_over_internet, *client.get_tools()]
        tool_node = ToolNode(tools)

        model_with_tools = model.bind_tools(tools)

        async def conversation(state: MessagesState):

            messages = state["messages"]

            response = await model_with_tools.ainvoke(
                [
                    # SystemMessage(content=sql_prompt),
                    *messages
                ]
            )

            return {"messages": [response]}

        def should_continue(state: MessagesState):
            messages = state["messages"]
            last_message = messages[-1]
            if last_message.tool_calls:
                return "tools"
            return END

        workflow = StateGraph(MessagesState)
        workflow.add_node("conversation", conversation)
        workflow.add_node("tools", tool_node)

        # Set the entrypoint as conversation
        workflow.add_edge(START, "conversation")
        workflow.add_edge("tools", "conversation")
        workflow.add_conditional_edges("conversation", should_continue, ["tools", END])

        yield workflow.compile()
