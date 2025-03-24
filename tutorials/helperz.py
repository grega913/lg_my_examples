
from icecream import ic
import streamlit as st

from langchain_core.messages import AIMessage, ToolMessage




def getGraphsSnapshot(graph, config):
    ic("def getGraphsSnapshot")
    snapshot = graph.get_state(config)
    setSessionSnapshot(snapshot=snapshot)
    return snapshot


# this message is displayed in case next in snapshot is "human"
# we also change session_state for human_response to True
def displayMessageAndChangeSessionStateIfNextIsHuman(snapshot):
    ic("def displayMessageAndChangeSessionStateIfNextIsHuman")
    if snapshot.next[0] == "human":
        st.session_state['human_response']=True
        st.write("The graph state is indeed interrupted before the 'human' node. We can act as the \"expert\" in this scenario and manually update the state by adding a new ToolMessage with our input.\n\nNext, respond to the chatbot's request by:\n1. Creating a ToolMessage with our response. This will be passed back to the chatbot.\n2. Calling update_state to manually update the graph state.", 
        unsafe_allow_html=True)
       


def updateGraphWithHumanResponse(graph, config, snapshot, update_message):
    ic("def updateGraphWithHumanResponse")
    ic(config)
    ic(snapshot)


    ai_message = snapshot.values["messages"][-1]


    human_response = (
        "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
        " It's much more reliable and extensible than simple autonomous agents."
    )

    if update_message and update_message!="":
        human_response = update_message


    tool_message = create_response(human_response, ai_message)
    graph.update_state(config, {"messages": [tool_message]})



# Next, create the "human" node. /
# This node function is mostly a placeholder in our graph that will trigger an interrupt./
# If the human does not manually update the state during the interrupt, it inserts a tool message so the LLM knows the user was requested but didn't respond. /
# This node also unsets the ask_human flag so the graph knows not to revisit the node unless further requests are made.
def create_response(response: str, ai_message: AIMessage):
    ic("def create_response")
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )



def setSessionSnapshot(snapshot):
    ic(f"setSessionSnapshot to {snapshot} ")
    st.session_state['snapshot'] = snapshot




def resumeGraphWithNone(graph, config):
    ic("def resumeGraphWithNone")
    events = graph.stream(None, config, stream_mode="values")
    return events



def showGraphsHistory(graph, config):
    ic("def showGraphsHistory")

    to_replay = None
    for state in graph.get_state_history(config):
        ic("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
        print("-" * 80)
        if len(state.values["messages"]) == 6:
            # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
            to_replay = state





