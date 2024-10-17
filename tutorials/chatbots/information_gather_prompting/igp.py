# https://langchain-ai.github.io/langgraph/tutorials/chatbots/information-gather-prompting/
# https://colab.research.google.com/drive/1pq3vViZposGXhjX32SLkLJKiyb5xbkLj

import streamlit as st
from icecream import ic
from typing import Annotated, List, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

import uuid

from pydantic import BaseModel

from tutorials.chatbots.helperz import cs_input_field




import os
from dotenv import load_dotenv
load_dotenv()

template = """Your job is to get information from a user about what type of prompt template they want to create.

You should get the following information from them:

- What the objective of the prompt is
- What variables will be passed into the prompt template
- Any constraints for what the output should NOT do
- Any requirements that the output MUST adhere to

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

After you are able to discern all the information, call the relevant tool."""

def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages


class PromptInstructions(BaseModel):
    """Instructions on how to prompt the LLM."""

    objective: str
    variables: List[str]
    constraints: List[str]
    requirements: List[str]

llm = ChatOpenAI()
llm_with_tool = llm.bind_tools([PromptInstructions])    


def info_chain(state):
    messages = get_messages_info(state["messages"])
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}



# New system prompt
prompt_system = """Based on the following requirements, write a good prompt template:

{reqs}"""

# Function to get the messages for the prompt
# Will only get messages AFTER the tool call
def get_prompt_messages(messages: list):
    tool_call = None
    other_msgs = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_call = m.tool_calls[0]["args"]
        elif isinstance(m, ToolMessage):
            continue
        elif tool_call is not None:
            other_msgs.append(m)
    return [SystemMessage(content=prompt_system.format(reqs=tool_call))] + other_msgs


def prompt_gen_chain(state):
    messages = get_prompt_messages(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


def get_state(state):
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    elif not isinstance(messages[-1], HumanMessage):
        return END
    return "info"

# Function to get the messages for the prompt
# Will only get messages AFTER the tool call
def get_prompt_messages(messages: list):
    tool_call = None
    other_msgs = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_call = m.tool_calls[0]["args"]
        elif isinstance(m, ToolMessage):
            continue
        elif tool_call is not None:
            other_msgs.append(m)
    return [SystemMessage(content=prompt_system.format(reqs=tool_call))] + other_msgs



@st.cache_resource
def createGraphIgp():
    ic("def createGraph")
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    memory = MemorySaver()
    workflow = StateGraph(State)

    workflow.add_node("info", info_chain)
    workflow.add_node("prompt", prompt_gen_chain)

    @workflow.add_node
    def add_tool_message(state: State):
        ic("def add_tool_message")
        return {
            "messages": [
                ToolMessage(
                    content="Prompt generated!",
                    tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                )
            ]
        }


    workflow.add_conditional_edges("info", get_state, ["add_tool_message", "info", END])
    workflow.add_edge("add_tool_message", "prompt")
    workflow.add_edge("prompt", END)
    workflow.add_edge(START, "info")
    graph = workflow.compile(checkpointer=memory)

    return graph



def streamGraphIgp(graph, user_input, config):
    ic(f"user_input in streamGraphIgp: {user_input}")
    output = None
    for output in graph.stream(
        {"messages": [HumanMessage(content=user_input)]}, config=config, stream_mode="updates"
    ):
        last_message = next(iter(output.values()))["messages"][-1]
        
        st.write(last_message)
        #last_message.pretty_print()



def igp():

    ic("def igp")
    graph = createGraphIgp()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    if graph:
        try:
            graph_image = graph.get_graph(xray=True).draw_mermaid_png()
        except Exception:
            graph_image = None
        st.image(graph_image, caption="Customizing State", width=200, use_column_width=False)


    st.title("Chatbots: Prompt Generation from User Requirementst")
    st.markdown("Learn more about Prompt Generation from User Requirements: https://langchain-ai.github.io/langgraph/tutorials/chatbots/information-gather-prompting")
    st.markdown("It is important to cache resource when defining a graph, because we do not want to initialize a graph with every call - it might lose memory.")
    
    with st.expander("How to use this"):
        st.text('''
                Just make a couple of statements and press Enter. The model should create appropriate prompt based on the questions.
                "hi!", "rag prompt", "1 rag, 2 none, 3 no, 4 no", "red"
        ''')

    users_input = cs_input_field()

    if users_input:
        st.markdown(users_input)
        streamGraphIgp(graph=graph, user_input=users_input, config=config)



# test without UI
'''
if __name__=="__main__":

    graph=createGraphIgp()
    import uuid

    cached_human_responses = ["hi!", "rag prompt", "1 rag, 2 none, 3 no, 4 no", "red", "q"]
    cached_response_index = 0
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    while True:
        try:
            user = input("User (q/Q to quit): ")
        except:
            user = cached_human_responses[cached_response_index]
            cached_response_index += 1
        print(f"User (q/Q to quit): {user}")
        if user in {"q", "Q"}:
            print("AI: Byebye")
            break
        output = None
        for output in graph.stream(
            {"messages": [HumanMessage(content=user)]}, config=config, stream_mode="updates"
        ):
            last_message = next(iter(output.values()))["messages"][-1]
            last_message.pretty_print()

        if output and "prompt" in output:
            print("Done!")

'''