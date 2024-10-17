# lg_my_examples

LangGraph Tutorials and HowTos with Streamlit

# run: streamlit run app.py

## 20241017

- Logic in streamlit seems to often be:

1.  create a function that returns a compiled graph and decor it with @st.cache_resource
2.  create a stream function with user_input

## 20241017

- Left Tutorials/Chatbots/CustomersSupport for later

## 20241016

- Tutorials/helperz.py: contains some reusable functions for langgraph

- Tutorials/quick_start/part 6: it is important to store snapshot in the streamlit session, because we need it to use it in later functions - at update

## 20241016

- Tutorials/quick_start finished for all 7 parts

## 20241014

Make sure to use @st.cache.resource when defining a graph, since a State object is initialized within.
This is relevant for all parts in Tutorials/quick_start.

## 20241015

In Tutorials/quick_start/part4 - issues with defining when a function on btn click should be executed. Solution with the use of st.session_state
