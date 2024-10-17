# https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-5-manually-updating-the-state

import streamlit as st
from icecream import ic
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage




import os
from dotenv import load_dotenv
load_dotenv()



# region Part 5: Manually Updating the State

@st.cache_resource
def createGraph():
    ic("def createGraph")
    class State(TypedDict):
        messages: Annotated[list, add_messages]


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
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    memory = MemorySaver()

    graph = graph_builder.compile(
        checkpointer=memory,
        # This is new!
        interrupt_before=["tools"],
        # Note: can also interrupt **after** actions, if desired.
        # interrupt_after=["tools"]
    )



    return graph

def stream_graph(graph, user_input, config):
    ic("stream_graph")

    #user_input = "I'm learning LangGraph. Could you do some research on it for me?"
    #config = {"configurable": {"thread_id": "1"}}

    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream({"messages": [("user", user_input)]}, config)
    for event in events:
        #ic(event)
        if "messages" in event:
            ic(event["messages"][-1])

def getGraphsSnapshot(graph, config):
    ic("def getGraphsSnapshot")
    snapshot = graph.get_state(config)
    return snapshot

def getExistingMessageFromSnapshot(snapshot):
    ic("def getExistingMessageFromSnapshot")
    existing_message = snapshot.values["messages"][-1]
    return existing_message

# we are updating state in graph with providing our answer
# https://colab.research.google.com/drive/1PXrCpvjmLrUKaj61O9ViWiEkfd5NBV7w#scrollTo=IkoPPlbUNYcv&line=1&uniqifier=1
def updateGraphState(graph, config, existing_message, my_answer):
    ic("def updateGraphState")
    ic(config)

  
    answer = (
            "LangGraph is a library for building stateful, multi-actor applications with LLMs. I am pretty sure of that"
        )

    if my_answer is not None and my_answer !="":
        answer = my_answer

    new_messages = [
        # The LLM API expects some ToolMessage to match its tool call. We'll satisfy that here.
        ToolMessage(content=answer, tool_call_id=existing_message.tool_calls[0]["id"]),
        # And then directly "put words in the LLM's mouth" by populating its response.
        AIMessage(content=answer),
    ]

    ic(new_messages[-1])



    graph.update_state(
        # Which state to update
        config,
        # The updated values to provide. The messages in our `State` are "append-only", meaning this will be appended
        # to the existing state. We will review how to update existing messages in the next section!
        {"messages": new_messages},
    )

    ic("Last 2 messages")
    ic(graph.get_state(config).values["messages"][-2:])




# second variant to update graph State, where we explicitly define the node - - in this example, the node is "chatbot"

def updateGraphState_v2(graph, config, my_answer):

    ic("def updateGraphsState_v2")

    answer = (
        "I 'am an AI Expert"
    )

    if my_answer is not None and my_answer !="":
        answer = my_answer



    graph.update_state(
        config,
        {"messages": [AIMessage(content=answer)]},
        # Which node for this function to act as. It will automatically continue
        # processing as if this node just ran.
        as_node="chatbot",
    )

    ic("Last 2 messages")
    ic(graph.get_state(config).values["messages"][-2:])


def part5():

    if 'existing_message' not in st.session_state:
        ic("setting the value of existing message in st.session_state")
        st.session_state['existing_message'] = ""




    graph = createGraph()
    config = {"configurable": {"thread_id": "1"}}

    if graph:
        try:
            graph_image = graph.get_graph(xray=True).draw_mermaid_png()
        except Exception:
            graph_image = None
        st.image(graph_image, caption="Manually updating State", width=250, use_column_width=False)





    st.title("Part 5: Manually Updating the State")
    st.markdown("[Learn more about Part 5: Manually Updating the State](https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-5-manually-updating-the-state)")
    st.markdown("It is important to cache resource when defining a graph, because we do not want to initialize a graph with every call - it might lose memory.")


    user_input = st.text_input("Enter your message:")

    if st.button("Submit"):
        if user_input and graph:
            ic("Sumbitted")
            st.markdown(f'<p style="color:blue;">User: {user_input}</p>', unsafe_allow_html=True)
            
            stream_graph(graph=graph, config=config, user_input=user_input)

            snapshot = getGraphsSnapshot(graph=graph, config=config)

            if snapshot:
                ic("we have snapshot and display it here")
                st.markdown(f"**Snapshot before update:**  {snapshot}")

            existingMessage = getExistingMessageFromSnapshot(snapshot)
            if existingMessage:
                ic("we have existingMessage and will display it here")
                ic("we have existingMessage and have stored it in st.session_state")

                st.session_state['existing_message'] = existingMessage
                st.markdown(f"ExistingMessage:  {existingMessage}")


    
    
    # Add text input field and button for updating state
    if st.session_state["existing_message"]:

        update_message = st.text_input("Enter your new message that will update state:")
        if st.button("Submit New Message"):
            ic(f"New message submitted: update_message {update_message}")
            
            st.markdown(f'<p style="color:blue;">New Message: {update_message}</p>', unsafe_allow_html=True)
            existing_message = st.session_state["existing_message"]
            

            #use v1 or v2

            #update the state with our answer - v1
            # updateGraphState(graph=graph, config=config, existing_message=existing_message, my_answer=update_message)


            #update the state with our answer -v2
            updateGraphState_v2(graph=graph, config=config, my_answer=update_message)


            
            snapshot = getGraphsSnapshot(graph=graph, config=config)
            st.markdown(f"**Snapshot after update:**  {snapshot}")

# endregion 



