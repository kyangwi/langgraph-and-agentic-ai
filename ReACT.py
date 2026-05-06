from typing import Annotated,Sequence,TypedDict
from dotenv import load_dotenv,find_dotenv
from langchain_core.messages import BaseMessage,ToolMessage,SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,END,START
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(find_dotenv())

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(a:int,b:int):
    """This is an addition function that adds two numbers together"""
    return a + b

@tool
def subtractor(a:int,b:int):
    """This is an addition function that adds two numbers together"""
    return a - b

@tool
def multiplier(a:int,b:int):
    """Multiply two numbers"""
    return a * b

@tool
def divider(a:int,b:int):
    """Divide number a/b """
    return a / b

tools = [add,subtractor,multiplier,divider]

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
).bind_tools(tools)


def model_call(state:AgentState) -> AgentState:
     
     system_prompt = SystemMessage(content="You are my AI assistant, please answerr my question to the best of your abilities")

     response = model.invoke([system_prompt] + state["messages"])

     return {"messages":[response]}

def should_continue(state:AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("agent",model_call)


tool_node = ToolNode(tools=tools)
graph.add_node("tools",tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue":"tools",
        "end":END
    }
)

graph.add_edge("tools","agent")

reACT = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages":[("user","add 40 + 12,add 6+8; and 4+9. then multiply the results")]}

print_stream(reACT.stream(inputs,stream_mode="values"))