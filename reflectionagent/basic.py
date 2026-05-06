from typing import List,Sequence,TypedDict,Annotated
from dotenv import load_dotenv,find_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END,StateGraph
from langgraph.graph.message import add_messages
from chains import reflection_chain,generation_chain

load_dotenv(find_dotenv())


class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]

graph = StateGraph(AgentState)

REFLECT = "reflect"
GENERATE = "generate"

def generate_node(state:AgentState):
    response =  generation_chain.invoke({
        "messages":state["messages"]
    })
    print("=======================================Generator================================================")
    print(response.content)
    return {"messages":[response]}

def reflect_node(state:AgentState):
    response =  reflection_chain.invoke(
        {
            "messages":state["messages"]
        }
    )
    print("=======================================Reflector================================================")
    print(response.content)
    return {"messages":[HumanMessage(content=response.content)]}

def should_continue(state):
    if len(state["messages"]) > 4:
        return "end"
   
    return "continue"

graph.add_node(GENERATE,generate_node)
graph.add_node(REFLECT,reflect_node)

graph.set_entry_point(GENERATE)

graph.add_conditional_edges(GENERATE,should_continue,{
    "end":END,
    "continue":REFLECT
})
graph.add_edge(REFLECT,GENERATE)

tweet = graph.compile()



tweet.invoke({"messages":[HumanMessage(content="I have built a chatbot consolidating data from multiple sources with trino and serve")]})





