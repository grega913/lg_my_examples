# https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/
# https://colab.research.google.com/drive/1sH5UFjL_veWriELyJSd5CNVaov1QsD-N#scrollTo=bIsME1kmwt_E

import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from icecream import ic
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from tutorials.chatbots.helperz import cs_input_field


import os
from dotenv import load_dotenv
load_dotenv()






# endregion

def customer_support():
    ic("def customer_support")
    st.title("Chatbots: Build a Customer Support Bot")
    st.markdown("Learn more about Customer Support Bot: https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/")
    
    st.markdown("It is important to cache resource when defining a graph, because we do not want to initialize a graph with every call - it might lose memory.")

    
    users_input = cs_input_field()

    if users_input:
        st.markdown(users_input)
    

