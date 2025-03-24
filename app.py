import streamlit as st

from tutorials.quick_start.part1 import part1
from tutorials.quick_start.part2 import part2
from tutorials.quick_start.part3 import part3
from tutorials.quick_start.part4 import part4
from tutorials.quick_start.part5 import part5
from tutorials.quick_start.part6 import part6
from tutorials.quick_start.part7 import part7
from tutorials.chatbots.customer_support.customer_support import customer_support
from tutorials.chatbots.information_gather_prompting.igp import igp
from tutorials.chatbots.code_assistant.code_assistant import code_assistant
from tutorials.rag.adaptive_rag.adaptive_rag import adaptive_rag
from playground import fixed_input_on_bottom


import os
from dotenv import load_dotenv
load_dotenv()


def intro():
    st.title("Welcome to LangGraph Lessons")
    st.sidebar.success("Select a demo above.")

st.set_page_config(page_title="LangGraph examples by GS", page_icon="🌶️", layout="wide")

page_names_to_funcs = {
    "—": intro,
    "Playground": fixed_input_on_bottom, 
    "Quick Start - Part 1: Build a Basic Chatbot" : part1,
    "Quick Start - Part 2: Enhancing the Chatbot with Tools": part2,
    "Quick Start - Part 3: Adding Memory to the Chatbot": part3,
    "Quick Start - Part 4: Human in the loop": part4,
    "Quick Start - Part 5: Manually Updating the State": part5,
    "Quick Start - Part 6: Customizing State": part6,
    "Quick Start - Part 7: Time Travel": part7,
    "Chatbots - Customer Support": customer_support,
    "Chatbots - IGP": igp,
    "Chatbots - Code Assistant": code_assistant,
    "RAG - Adaptiv_RAG:": adaptive_rag

}

demo_name = st.sidebar.selectbox("Choose a part", page_names_to_funcs.keys(), index=1)
page_names_to_funcs[demo_name]()





