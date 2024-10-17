# https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-7-time-travel
# https://colab.research.google.com/drive/1PXrCpvjmLrUKaj61O9ViWiEkfd5NBV7w#scrollTo=F5xyq6KtPFHt&line=1&uniqifier=1

import streamlit as st
from icecream import ic
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
#from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel
import time

from tutorials.helperz import getGraphsSnapshot, showGraphsHistory, create_response, updateGraphWithHumanResponse, resumeGraphWithNone

import os
from dotenv import load_dotenv
load_dotenv()


# region Part 7: Time Travel
#https://colab.research.google.com/drive/1PXrCpvjmLrUKaj61O9ViWiEkfd5NBV7w#scrollTo=-FSX2DzzPZ0P&line=1&uniqifier=1

@st.cache_resource
def createGraphPart7():
    ic("def createGraphPart7")

    class State(TypedDict):
        ic("class State(TypeDict)")
        messages: Annotated[list, add_messages]
        # This flag is new
        ask_human: bool


    class RequestAssistance(BaseModel):
        ic("class RequestAssistance(BaseModel)")

        """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.

        To use this function, relay the user's 'request' so the expert can provide the right guidance.
        """

        request: str
    

    tool = TavilySearchResults(max_results=2)
    tools = [tool]
    llm = ChatOpenAI()
    # We can bind the llm to a tool definition, a pydantic model, or a json schema
    llm_with_tools = llm.bind_tools(tools + [RequestAssistance])

    def chatbot(state: State):
        ic("def chatbot")

        response = llm_with_tools.invoke(state["messages"])
        ask_human = False
        if (
            response.tool_calls
            and response.tool_calls[0]["name"] == RequestAssistance.__name__
        ):
            ask_human = True
        return {"messages": [response], "ask_human": ask_human}


    graph_builder = StateGraph(State)

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(tools=[tool]))

    def human_node(state: State):
        ic("def human_node")

        new_messages = []
        if not isinstance(state["messages"][-1], ToolMessage):
            # Typically, the user will have updated the state during the interrupt.
            # If they choose not to, we will include a placeholder ToolMessage to
            # let the LLM continue.
            new_messages.append(
                create_response("No response from human.", state["messages"][-1])
            )
        return {
            # Append the new messages
            "messages": new_messages,
            # Unset the flag
            "ask_human": False,
        }
    
    graph_builder.add_node("human", human_node)

    def select_next_node(state: State):
        ic("def select_next_node")

        if state["ask_human"]:
            return "human"
        # Otherwise, we can route as before
        return tools_condition(state)

    graph_builder.add_conditional_edges(
        "chatbot",
        select_next_node,
        {"human": "human", "tools": "tools", "__end__": "__end__"},
    )

    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge("human", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    memory = MemorySaver()

    graph = graph_builder.compile(
        checkpointer=memory,
        interrupt_before=["human"],
    )

    return graph


def streamGraphPart7(graph, user_input, config):
    ic(user_input)
    events = graph.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )
    for event in events:
        if "messages" in event:
            ic(event["messages"][-1])
            last_message = event["messages"][-1]
    
    return last_message




def toast():
    msg = st.toast('we have changed state and can use it to time travel')
    time.sleep(2)
    msg.toast('Ready!', icon = "🥞")



def part7():

    ic("def part7")

    # we need to implement a var in st.session_state for human_response
    # if this is something, then we display a user_input field where we expect to collect something
    
    
    if 'stored_state' not in st.session_state:
        ic("setting the value of human_response initially to False in st.session_state")
        st.session_state['stored_state'] = False
    
    '''
    if 'snapshot' not in st.session_state:
        ic("setting the value of snapshot initially to None in st.session_state")
        st.session_state['snapshot'] = None
    '''


    
    graph = createGraphPart7()
    config = {"configurable": {"thread_id": "1"}}

    if graph:
        try:
            graph_image = graph.get_graph(xray=True).draw_mermaid_png()
        except Exception:
            graph_image = None
        st.image(graph_image, caption="Customizing State", width=250, use_column_width=False)
    

    st.title("Part 7: Time Travel")
    st.markdown("Learn more about Part 7: Time travel: https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-7-time-travel")
    st.markdown("It is important to cache resource when defining a graph, because we do not want to initialize a graph with every call - it might lose memory.")

    st.markdown("ask me a couple of questions")

    user_input = st.text_input("Enter your message: - I need some expert guidance for building this AI agent. Could you request assistance for me?")

    

    if st.button("Submit"):
        ic(user_input)
        to_replay = None
        
        last_message = streamGraphPart7(graph=graph, user_input=user_input, config=config)
        
        
        st.write(" - * - * - * - * - * - * - * -")
        for state in graph.get_state_history(config):
            
            if len(state.values["messages"]) < 6:
                st.markdown("Num Messages: " + str(len(state.values["messages"])) +  " Next: " + str(state.next))
                st.markdown("----------------------------------------")

            if len(state.values["messages"]) == 6:
                # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
                to_replay = state
                st.session_state["stored_state"] = state

    if st.session_state["stored_state"]:
        toast()

        currState = st.session_state["stored_state"]

        ic("after toast")
        st.markdown(currState.next)
        st.markdown(currState.config)

        for event in graph.stream(None, to_replay.config, stream_mode="values"):
            if "messages" in event:
                st.markdown(str(event["messages"][-1]))


  


# end region
