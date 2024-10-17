import streamlit as st
from datetime import datetime
from icecream import ic


def fixed_input_on_bottom():
    ic("def fixed_input_on_bottom")

    with st.container():
        st.markdown(
            """
            <style>
            div[data-testid="stAppViewContainer"]{
                position:fixed;
                bottom:18%;
                padding: 10px;
            }
            div[data-testid="stForm"]{
                position:fixed;
                right:10%;
                left: 10%;
                bottom: 2%;
                border: 2px solid green;
                padding: 10px;
                z-index: 1;
            }
            </style>
            """, unsafe_allow_html=True
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = list()

        with st.form("input_form"):
            "*Enter messages here*"

            col1, col2 = st.columns([10, 1])

            with col1:
                message = st.text_input("message", label_visibility="collapsed")
                
                if message:    
                    st.session_state.chat_history.append(f"{datetime.now().strftime(r'%H:%M:%S')}:  {message}")

            with col2:
                submitted = st.form_submit_button(use_container_width=True)

        if submitted:
            for msg in st.session_state.chat_history:
                with st.chat_message("user", avatar="🤪"):
                    st.write("bla" + str(msg))