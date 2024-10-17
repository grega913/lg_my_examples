#https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-3-adding-memory-to-the-chatbot


import streamlit as st
from icecream import ic



from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_groq import ChatGroq
from IPython.display import Image, display

from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition

import matplotlib.pyplot as plt

from langgraph.checkpoint.memory import MemorySaver

import os
from dotenv import load_dotenv
load_dotenv()

# region Part 3: Adding Memory to the Chatbot

@st.cache_resource
def simple_chatbot_with_tools_and_memory():

    ic("simple_chatbot_with_tools_and_memory")

    class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
        messages: Annotated[list, add_messages]



    memory = MemorySaver()

    graph_builder = StateGraph(State)


    tool = TavilySearchResults(max_results=2)
    tools = [tool]

    llm = ChatGroq()
    llm_with_tools = llm.bind_tools(tools)


    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}


    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    graph = graph_builder.compile(checkpointer=memory)

    try:
        graph_image = graph.get_graph(xray=True).draw_mermaid_png()
    except Exception:
        graph_image = None

    return graph, graph_image


def stream_graph_with_tools_and_memory_updates(user_input: str):

    ic("stream_graph_with_tools_and_memory_updates", user_input)

    config = {"configurable": {"thread_id": "1"}}

    graph, graph_image = simple_chatbot_with_tools_and_memory()

    if graph_image:
        st.image(graph_image, caption="Chatbot with Tools and Memory Graph", width=200, use_column_width=False)
    '''
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            ic("Assistant:", value["messages"][-1].content)
            response = value["messages"][-1].content
            st.write(f"Assistant: {response}")
    '''

    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )

    for event in events:
        # ic(event)
        #st.write(f"Assistant: {event}")
        st.write(event["messages"][-1])


def part3():
    st.title("Part 3: Adding Memory to the Chatbot")
    st.markdown("[Learn more about Part 3: Adding Memory to the Chatbot](https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-3-adding-memory-to-the-chatbot)")
    st.markdown("It is important to cache resource when defining a graph, because we do not want to initialize a graph with every call - it might lose memory.")
    
    user_input = st.text_input("Enter your message:")
    if st.button("Submit"):
        if user_input:
            st.markdown(f'<p style="color:blue;">User: {user_input}</p>', unsafe_allow_html=True)
            stream_graph_with_tools_and_memory_updates(user_input)

# endregion  