from langchain_chat import vectordb, chatbot
import streamlit as st

st.title("Q&A Chatbot")
btn = st.button("Prepare Things!")
if btn:
    with st.spinner("Preparing things..."):
        vectordb()

question = st.text_input("Question:")

if question:
    chain = chatbot()
    response = chain(question)
    
    st.header("Answer:")
    st.write(response["result"])

