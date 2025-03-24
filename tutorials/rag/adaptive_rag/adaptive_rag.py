# https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/


# this is the main file for adaptive rag tutorial



import streamlit as st
from icecream import ic
from typing import Annotated, List, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image
import uuid
from pydantic import BaseModel, Field
from pprint import pprint

from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langgraph.graph import END, StateGraph, START

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

import os
from dotenv import load_dotenv
load_dotenv()

from tutorials.rag.adaptive_rag.compile_graph import compile_graph



# region adaptiveRag

web_search_tool = TavilySearchResults(k=3)



     





# endregion





     


def adaptive_rag():
        




        with st.container():
            st.title("Adaptive RAG")
            st.image(image=Image.open("static/adaptive_rag.png"), caption="Graph Code Assistant", width=800)
            
            st.markdown("Learn more about Adaptive RAG: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/")
            st.markdown("It is important to cache resource when defining a graph, because we do not want to initialize a graph with every call - it might lose memory.")
            
            with st.expander("How to use this"):
                st.write('''
                        Just make a couple of statements and press Enter.\n
                        Detailed instructions here: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/
                        ''')



if __name__== "__main__":
    ic("def main in adaptive_rag")
    app = compile_graph()



    '''
    ic(question_router.invoke({"question": "Who will the Bears draft first in the NFL draft?"}))
    ic(question_router.invoke({"question": "What are the types of agent memory?"}))
    '''






    '''
    # Run
    inputs = {
        "question": "What player at the Bears expected to draft first in the 2024 NFL draft?"
    }
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n"),
    '''

