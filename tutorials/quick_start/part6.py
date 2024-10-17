# https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-6-customizing-state

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

from tutorials.helperz import getGraphsSnapshot, displayMessageAndChangeSessionStateIfNextIsHuman, create_response, updateGraphWithHumanResponse, resumeGraphWithNone

import os
from dotenv import load_dotenv
load_dotenv()




# region Part 6: Customizing State


# main function that creates a graph - the whole code from colab is encapsulated in a function here
#https://colab.research.google.com/drive/1PXrCpvjmLrUKaj61O9ViWiEkfd5NBV7w#scrollTo=i8Dw2KUVOYuC&line=1&uniqifier=1
@st.cache_resource
def createGraph():
    ic("def createGraph - Part 6")

    class State(TypedDict):
        messages: Annotated[list, add_messages]
        # This flag is new
        ask_human: bool

    #Next, define a schema to show the model to let it decide to request assistance.
    class RequestAssistance(BaseModel):
        """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.

        To use this function, relay the user's 'request' so the expert can provide the right guidance.
        """

        request: str

    # define tools, model, and bind tools to a model
    tool = TavilySearchResults(max_results=2)
    tools = [tool]
    llm = ChatOpenAI()
    # We can bind the llm to a tool definition, a pydantic model, or a json schema
    llm_with_tools = llm.bind_tools(tools + [RequestAssistance])

    # Next, define the chatbot node. The primary modification here is flip the ask_human flag if we see that the chat bot has invoked the RequestAssistance flag.
    def chatbot(state: State):
        response = llm_with_tools.invoke(state["messages"])
        ask_human = False
        if (
            response.tool_calls
            and response.tool_calls[0]["name"] == RequestAssistance.__name__
        ):
            ask_human = True
        return {"messages": [response], "ask_human": ask_human}
    
    # Next, create the graph builder and add the chatbot and tools nodes to the graph, same as before.
    graph_builder = StateGraph(State)

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(tools=[tool]))


    
    def human_node(state: State):
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


    # Next, define the conditional logic. The select_next_node will route to the human node if the flag is set./
    # Otherwise, it lets the prebuilt tools_condition function choose the next node.

    # Recall that the tools_condition function simply checks to see if the chatbot has responded with any tool_calls in its response message./
    # If so, it routes to the action node. Otherwise, it ends the graph.
    def select_next_node(state: State):
        if state["ask_human"]:
            return "human"
        # Otherwise, we can route as before
        return tools_condition(state)
    
    graph_builder.add_conditional_edges(
        "chatbot",
        select_next_node,
        {"human": "human", "tools": "tools", "__end__": "__end__"},
    )

    # Finally, add the simple directed edges and compile the graph./
    # These edges instruct the graph to always flow from node a->b whenever a finishes executing.

    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge("human", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    memory = MemorySaver()

    graph = graph_builder.compile(
        checkpointer=memory,
        interrupt_before=["human"],
    )

    return graph

def streamGraph(graph, user_input, config):
    ic( "def streamGraph")
    events = graph.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )
    for event in events:
        if "messages" in event:
            ic(event["messages"][-1])


# this is displayed in streamlit page
def part6():
    ic("def part6")
    
    # we need to implement a var in st.session_state for human_response
    # if this is something, then we display a user_input field where we expect to collect something
    if 'human_response' not in st.session_state:
        ic("setting the value of human_response initially to False in st.session_state")
        st.session_state['human_response'] = False
    
    if 'snapshot' not in st.session_state:
        ic("setting the value of snapshot initially to None in st.session_state")
        st.session_state['snapshot'] = None



    graph = createGraph()
    config = {"configurable": {"thread_id": "1"}}

    if graph:
        try:
            graph_image = graph.get_graph(xray=True).draw_mermaid_png()
        except Exception:
            graph_image = None
        st.image(graph_image, caption="Customizing State", width=250, use_column_width=False)
    

    st.title("Part 6: Customizing State")
    st.markdown("Learn more about Part 6: Customizing State: https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-6-customizing-state")
    st.markdown("It is important to cache resource when defining a graph, because we do not want to initialize a graph with every call - it might lose memory.")


    user_input = st.text_input("Enter your message: - I need some expert guidance for building this AI agent. Could you request assistance for me?")

    if st.button("Submit"):
        ic(user_input)
        streamGraph(graph=graph, user_input=user_input, config=config)
        snapshotBeforeUpdate = getGraphsSnapshot(graph=graph, config=config)
        
        if snapshotBeforeUpdate:
            ic("we have snapshotBeforeUpdate and display it here")
            st.markdown(snapshotBeforeUpdate.next)
            displayMessageAndChangeSessionStateIfNextIsHuman(snapshotBeforeUpdate)
            st.markdown(f"**Snapshot before update:**  {snapshotBeforeUpdate}")

    # add text_input for collecting human_response
    # this field is displayed only if the session_state (Stremlit) for human_message == True
    if st.session_state["human_response"]:
        ic("in the part 6 block where we have human_response")
        snapshot_from_state = st.session_state["snapshot"]
        update_message = st.text_input("Enter your new message. Something smart.")
        if st.button("Submit New Message"):
            

            ic(f"New message submitted: {update_message}")
            ic(f"Snapshot_from_state: {snapshot_from_state}")
            # update graph with Human Response:
            updateGraphWithHumanResponse(graph=graph, snapshot=snapshot_from_state, config=config, update_message=update_message)

            # here we check the snapshot after human response
            snapshotAfterHumanresponse = getGraphsSnapshot(graph=graph, config=config)
            if snapshotAfterHumanresponse:
                st.markdown(f"**Snapshot after human response:**  {snapshotAfterHumanresponse}")
                
                # here we continue/resume the graph execution 
                events = resumeGraphWithNone(graph=graph, config=config)
                st.markdown("**Aftrer resumeGraphWithNone:**")
                for event in events:
                    if "messages" in event:
                        st.write(event["messages"][-1])






# endregion