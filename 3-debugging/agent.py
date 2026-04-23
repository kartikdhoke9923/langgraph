from typing import Annotated
from typing_extensions import TypedDict
# from langchain_openai import ChatOpenAI
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

import os
from dotenv import load_dotenv
load_dotenv() # to load env variables from .env file

os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')
os.environ["LANGSMITH_API_KEY"]=os.getenv('LANGSMITH_API_KEY')

os.environ["LANGSMITH_TRACING"]="true"  # from langchain docs and true is not booloean
os.environ["LANGSMITH_PROJECT"]="Test_Project"

from langchain.chat_models import init_chat_model
llm=init_chat_model('groq:llama-3.1-8b-instant')

class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def make_tool_graph():
    ## state graph with tool call
    from langchain_core.tools import tool

    @tool
    def add(a:float, b:float) -> float:
        """Add two numbers together."""
        return a + b

    tools=[add]
    tool_node = ToolNode([add])

    llm_with_tool=llm.bind_tools([add])

    def call_llm_model(state:State) -> State:
        return {"messages":[llm_with_tool.invoke(state["messages"])]}
    


    builder=StateGraph(State)
    builder.add_node("call_llm_model",call_llm_model)
    builder.add_node("tools", ToolNode(tools))

    # adding edges
    builder.add_edge(START, "call_llm_model")
    builder.add_conditional_edges("call_llm_model",
                                # if the latest message in the messages list contains a tool call, then go to the tools node
                                # if the latest message in the messages list does not contain a tool call, then go to the end node
                                tools_condition)
    builder.add_edge("tools", "call_llm_model")      

    graph=builder.compile()              
    from IPython.display import Image,  display
    display(Image(graph.get_graph().draw_mermaid_png()))

    return graph
tool_agent=make_tool_graph() #
  # run langgraph dev : it will check where the agennt in file is 