import os

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API key from .env
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

st.title("OOPSMate AI - LangChain Demo")

if not api_key:
    st.error("GOOGLE_API_KEY missing. Please check your .env file.")
    st.stop()

# Connect Gemini with LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=api_key
)

# Create a reusable prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are a friendly Java and OOPS teacher.

    Explain the given topic in simple English.

    Include:
    1. A simple definition.
    2. An everyday example.
    3. A Java code example.
    4. A simple explanation of the code.

    Topic: {topic}
    """
)

# Convert the AI response into plain text
output_parser = StrOutputParser()

# Connect prompt -> Gemini -> output parser
chain = prompt | llm | output_parser

topic = st.text_input("Enter a Java or OOPS topic")

if st.button("Explain Topic"):
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        with st.spinner("AI is preparing your explanation..."):
            try:
                answer = chain.invoke({"topic": topic})
                st.markdown(answer)
            except Exception as error:
                st.error(f"Something went wrong: {error}")