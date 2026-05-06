from dotenv import load_dotenv,find_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict,Annotated,Sequence
from langchain_core.messages import BaseMessage,SystemMessage,HumanMessage,ToolMessage
from operator import add as add_messages
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool 

load_dotenv(find_dotenv())


llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-pro",
    temperature=0
)

embeddings =GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")


pdf_path = "doc.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"Pdf file not found: {pdf_path}")

# loading the psf
pdf_loader = PyPDFLoader(pdf_path)

# loading the pages
pages = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,chunk_overlap = 200
)

splited_pages = text_splitter.split_documents(pages)

persit_directory = "./embedings"
collection_name = "stockmarket"

if not os.path.exists(persit_directory):
    os.makedirs(persit_directory)

try:
    vectorstore = Chroma.from_documents(
        documents=splited_pages,
        embedding=embeddings,
        persist_directory=persit_directory,
        collection_name=collection_name
    )
    print(f"created chromadb vector store")
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")

retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {
        "k":5
    }
)

@tool
def retriever_tool(query:str) -> str:
    """
    This tool searches and returns information from the Capital Markets Quarterly Bulletin for the quarter ended December 2024.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "I found no releavant information in the stock market perfomance 2024 document."
    results = []
    for i,doc in enumerate(docs):
        results.append(f"Document: {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)
    
tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]

def should_continue(state:AgentState):
    """ CHECK IF THE LAST MESSAGE CONTAINS TOOL CALLS."""

    result = state["messages"][-1]
    return hasattr(result,"tool_calls") and len(result.tool_calls) > 0 

system_prompt = """

    You are an intelligent AI assistant that answers questions about the Capital Markets Quarterly Bulletin for the quarter ended December 2024.
    The document describes Uganda's capital markets performance during the quarter and CMA's regulatory and supervisory activities.
    Use the retriever tool available to answer questions from the bulletin. You may make multiple calls if needed.
    If you need to look up supporting information before answering a follow-up question, you are allowed to do so.
    Please always cite the specific parts of the document you use in your answers.

"""
tools_dict = {our_tool.name: our_tool for our_tool in tools}


def call_llm(state:AgentState) -> AgentState:
    """
        Function to call the llm with the current state
    """

    messages = list(state["messages"])
    messages =[SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)

    return {"messages":[message]}

def take_action(state:AgentState):
    "Execute too calls from the llm's response"

    tool_calls = state["messages"][-1].tool_calls
    result = []

    for t in tool_calls:
        print(f"Calling Tool:{t['name']} with query: {t['args'].get('query','No query provided')} ")
        if not t['name'] in tools_dict: # checks if a valid tool is present
            print(f"\nTool:{t['name']} does not exist.")
            result.append(
                ToolMessage(
                    content="Incorrect Tool name, please Retry and select tool from List available tools",
                    tool_call_id=t["id"],
                )
            )
        else:
            tool_result = tools_dict[t['name']].invoke(t['args'].get('query',''))
            print(f"Result length: {len(str(tool_result))}")
            result.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=t["id"],
                )
            )
    print("Tools Execution complete. Back to the model")

    return {"messages":result}

graph = StateGraph(AgentState)

graph.add_node("llm",call_llm)
graph.add_node("retriever_agent",take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {
    True:"retriever_agent",
    False:END
     }
)

graph.add_edge("retriever_agent","llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

def running_agent():
    print("==============RAG Agent=============")
    while True:
        user_input = input("Enter your question")
        if user_input.lower() in ["exit","quit"]:
            break
        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages":messages})

        print("====================ANSWER====================")
        print(result["messages"][-1].content)

if __name__ == "__main__":
    running_agent()
