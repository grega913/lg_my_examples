# https://langchain-ai.github.io/langgraph/tutorials/code_assistant/langgraph_code_assistant/

import streamlit as st
from icecream import ic
from typing import Annotated, List, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import uuid
from pydantic import BaseModel, Field
from tutorials.chatbots.helperz import cs_input_field

from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langgraph.graph import END, StateGraph, START

import os
from dotenv import load_dotenv
load_dotenv()


from playground import fixed_input_on_bottom

# region CodeAssistant

def getDocs():
    ic("def getDocs")
    # LCEL docs
    url = "https://python.langchain.com/docs/concepts/#langchain-expression-language-lcel"
    loader = RecursiveUrlLoader(
        url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
    )
    docs = loader.load()

    # Sort the list based on the URLs and get the text
    d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
    d_reversed = list(reversed(d_sorted))
    concatenated_content = "\n\n\n --- \n\n\n".join(
        [doc.page_content for doc in d_reversed]
    )

    return concatenated_content

# Grader prompt
code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a coding assistant with expertise in LCEL, LangChain expression language. \n 
    Here is a full set of LCEL documentation:  \n ------- \n  {context} \n ------- \n Answer the user 
    question based on the above provided documentation. Ensure any code you provide can be executed \n 
    with all required imports and variables defined. Structure your answer with a description of the code solution. \n
    Then list the imports. And finally list the functioning code block. Here is the user question:""",
        ),
        ("placeholder", "{messages}"),
    ]
)

# Data model
class code(BaseModel):
    """Schema for code solutions to questions about LCEL."""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")



concatenated_content = getDocs()

expt_llm = "gpt-4o-mini"
llm = ChatOpenAI(temperature=0, model=expt_llm)
code_gen_chain_oai = code_gen_prompt | llm.with_structured_output(code)
question = "How do I build a RAG chain in LCEL?"
solution = code_gen_chain_oai.invoke(
    {"context": concatenated_content, "messages": [("user", question)]}
)
solution





# create/compile graph
def createGraphCodeAssistant():
    

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("generate", generate)  # generation solution
    workflow.add_node("check_code", code_check)  # check code
    workflow.add_node("reflect", reflect)  # reflect

    # Build graph
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "check_code")
    workflow.add_conditional_edges(
        "check_code",
        decide_to_finish,
        {
            "end": END,
            "reflect": "reflect",
            "generate": "generate",
        },
    )
    workflow.add_edge("reflect", "generate")
    graph = workflow.compile()

    return graph




# endregion


# main function for displaying streamlit
def code_assistant():

    ic("def code_assistant")

    with st.container():
        st.title("Code generation with RAG and self-correction")
        st.markdown("Learn more about Code Assistant: https://langchain-ai.github.io/langgraph/tutorials/code_assistant/langgraph_code_assistant/")
        st.markdown("It is important to cache resource when defining a graph, because we do not want to initialize a graph with every call - it might lose memory.")
        with st.expander("How to use this"):
            st.write('''
                    Just make a couple of statements and press Enter.\n
                    Detailed instructions here: https://langchain-ai.github.io/langgraph/tutorials/code_assistant/langgraph_code_assistant/
            ''')

        fixed_input_on_bottom()






