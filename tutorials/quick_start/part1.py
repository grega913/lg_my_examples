#https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-1-build-a-basic-chatbot

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

#graph.get_graph().print_ascii()




# region Part 1: Build a BasicChatbot



def simple_chatbot():

    class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
        messages: Annotated[list, add_messages]

    graph_builder = StateGraph(State)

    llm = ChatGroq()


    def chatbot(state: State):
        return {"messages": [llm.invoke(state["messages"])]}


    # The first argument is the unique node name
    # The second argument is the function or object that will be called whenever
    # the node is used.
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile()

    try:
        graph_image = graph.get_graph(xray=True).draw_mermaid_png()
    except Exception:
        graph_image = None

    return graph, graph_image

def stream_graph_updates(user_input: str):
    ic(user_input)

    graph, graph_image = simple_chatbot()
    if graph_image:
        st.image(graph_image, caption="Chatbot Graph", width=200, use_column_width=False)
    
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            ic("Assistant:", value["messages"][-1].content)
            response = value["messages"][-1].content
            st.write(f"Assistant: {response}")

def run_simple_chatbot():
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            response = ""
            graph = simple_chatbot()
            for event in graph.stream({"messages": [("user", user_input)]}):
                for value in event.values():
                    response = value["messages"][-1].content
                    st.markdown(f'<p style="color:green;">Assistant: {response}</p>', unsafe_allow_html=True)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break

def part1():
    st.title("Part 1: Build a Basic Chatbot")
    st.markdown("[Learn more about Part 1: Build a Basic Chatbot](https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-1-build-a-basic-chatbot)")
    
    user_input = st.text_input("Enter your message:")
    if st.button("Submit"):
        if user_input:
            st.markdown(f'<p style="color:blue;">User: {user_input}</p>', unsafe_allow_html=True)
            stream_graph_updates(user_input)

# endregion