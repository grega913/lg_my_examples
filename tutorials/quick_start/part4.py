# https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-4-human-in-the-loop


import streamlit as st
from icecream import ic


from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition



from langgraph.checkpoint.memory import MemorySaver

import os
from dotenv import load_dotenv
load_dotenv()

# region Part 4: Human in the loop



def testPrint(str):
    ic(str)


@st.cache_resource
def simple_chatbot_with_human_in_the_loop():

    ic("simple_chatbot_with_human_in_the_loop")

    memory = MemorySaver()

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    graph_builder = StateGraph(State)

    tool = TavilySearchResults(max_results=2)
    tools = [tool]

    #llm = ChatGroq()
    llm= ChatOpenAI()
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
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")



    graph = graph_builder.compile(
        checkpointer=memory,
        # This is new!
        interrupt_before=["tools"],
        # Note: can also interrupt __after__ tools, if desired.
        # interrupt_after=["tools"]
)

    try:
        graph_image = graph.get_graph(xray=True).draw_mermaid_png()
    except Exception:
        graph_image = None

    return graph, graph_image


'''
def stream_graph_with_human_in_the_loop(user_input: str):
    ic("stream_graph_with_human_in_the_loop", user_input)

    graph, graph_image = simple_chatbot_with_human_in_the_loop()
    

    if graph_image:
        
        st.image(graph_image, caption="Chatbot with Human In The Loop", width=200, use_column_width=False)
        
        config = {"configurable": {"thread_id": "1"}}

        events = graph.stream(
                {"messages": [("user", user_input)]}, config, stream_mode="values"
            )

        for event in events:
            if "messages" in event:
                ic(event["messages"][-1])


        snapshot = graph.get_state(config)
        
        st.markdown(f"Snapshot.next: {snapshot.next}")
        st.markdown(f"Snapshot: {snapshot}")

        values = snapshot.values
        ic(len(values))
        ic(values)

        for msg in values["messages"]:
            st.write(" - - - ")
            st.write(f"• {type(msg).__name__}: {msg.content}")  

        existing_message = snapshot.values["messages"][-1]

        if existing_message.tool_calls:
            proceed_button = st.button("Proceed", key="btnProceed")
            if proceed_button:
                testPrint("kr neki")
'''


def stream_graph_with_human_in_the_loop_and_graph(graph, user_input: str):
    ic("stream_graph_with_human_in_the_loop_and_graph", user_input)

    if 'existing_message_tool_calls' not in st.session_state:
        st.session_state["existing_message_tool_calls"] = None


    try:
        graph_image = graph.get_graph(xray=True).draw_mermaid_png()
    except Exception:
        graph_image = None
    

    if graph_image:
        st.image(graph_image, caption="Chatbot with Human In The Loop", width=200, use_column_width=False)
        config = {"configurable": {"thread_id": "1"}}

        events = graph.stream(
                {"messages": [("user", user_input)]}, config, stream_mode="values"
            )

        for event in events:
            if "messages" in event:
                ic(event["messages"][-1])


        snapshot = graph.get_state(config)

    
        st.session_state.snapshot = snapshot


        
        st.markdown(f"Snapshot.next: {snapshot.next}")
        st.markdown(f"Snapshot: {snapshot}")




        values = snapshot.values
        ic(len(values))
        ic(values)

        for msg in values["messages"]:
            st.write(" - - - ")
            st.write(f"• {type(msg).__name__}: {msg.content}")  

        existing_message = snapshot.values["messages"][-1]



        if existing_message.tool_calls:
            ic("existing_message has tool_calls, we will set the session_state...")
            st.session_state['existing_message_tool_calls'] = True



                    






# `None` will append nothing new to the current state, letting it resume as if it had never been interrupted
def streamWithNone(graph, config):
    ic("streamWithNone")
    events = graph.stream(None, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            st.write(event["messages"][-1])






def showSnapshotInState():
    ic("showSnapshotInState")
    if 'snapshot' not in st.session_state:
        ic("setting the value of snapshot in state to 'empty'")
        st.session_state['snapshot'] = "empty"
    
    st.write(f"**Snapshot from state:** {st.session_state.snapshot}")
        


def showSessionState():
    st.write(f"**Session_State:** {st.session_state}")    







def part4():

    if 'existing_message_tool_calls' not in st.session_state:
        ic("setting the value of snapshot in state to 'empty'")
        st.session_state['existing_message_tool_calls'] = False
        
    st.title("Part 4: Human-in-the-loop")
    st.markdown("[Learn more about Part 4: Human in the loop](https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-4-human-in-the-loop)")
    st.markdown("It is important to cache resource when defining a graph, because we do not want to initialize a graph with every call - it might lose memory.")

    graph, graph_image = simple_chatbot_with_human_in_the_loop()
    config = {"configurable": {"thread_id": "1"}}   

    
    user_input = st.text_input("Enter your message:")
    if st.button("Submit"):
        if user_input:
            st.markdown(f'<p style="color:blue;">User: {user_input}</p>', unsafe_allow_html=True)
            stream_graph_with_human_in_the_loop_and_graph(graph, user_input)
            
    
    
    if st.session_state["existing_message_tool_calls"]:
        if st.button("Proceed"):
            ic("Proceeeeed now")
            streamWithNone(graph=graph, config=config)

    showSessionState()
    
    


# endregion 