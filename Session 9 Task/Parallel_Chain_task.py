from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()

# ___________________ Model_1
model_1 = HuggingFacePipeline.from_model_id(
    model_id="distilgpt2",
    task="text-generation"
)

#___________ Model_2
model_2 = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash"
)

#______________ Model 3
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

model_3 = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=api_key,
    base_url=base_url
)


template_1 = PromptTemplate(
    template="""
You are a teacher. Read the following text and explain it in detail 
so that even a beginner can easily understand it.  
Text: {text}
""",
    input_variables=["text"]
)

template_2 = PromptTemplate(
    template="""
You are a quiz master. Based on the following text, do the following:
1. Create 3 Questions and their Answers.  
2. Create 3 Multiple-Choice Quiz Questions (4 options each, mark the correct answer).  
Text: {text}
""",
    input_variables=["text"]
)


template_3 = PromptTemplate(
    template="""
You are an analyst. Based on the following text, provide:  

1. A short **Summary** (5-6 lines).  
2. The **Overall Sentiment** (Positive / Negative / Neutral).  
3. The **Main Topic** of the text.  

Text: {text}
""",
    input_variables=["text"]
)


parser = StrOutputParser()

Explanation_chain = template_1 | model_1 | parser
Quiz_question_chain = template_2 | model_2 | parser
summary_sentiment_Chain = template_3 | model_3 | parser



final_chain = RunnableParallel({
    "explanation_chain" : Explanation_chain,
    "quiz_question_chain" : Quiz_question_chain,
    "sentiment_summary_chain" : summary_sentiment_Chain 
})



import streamlit as st
st.title("🌍 Multi-Model AI Climate App")

st.subheader("Enter your text for analysis")
user_text = st.text_area("Type or paste your text here:", height=200)

if st.button("Generate Results"):
    with st.spinner("Generative Result.... "):
        if user_text.strip() == "":
            st.warning("Please enter some text before generating results!")
        else:
            result = final_chain.invoke({"text": user_text})
            
            st.subheader("Explanation")
            st.write(result["explanation_chain"])

            st.subheader("Quiz and Q & A")
            st.write(result["quiz_question_chain"])

            st.subheader("Summary Sentiment Analysis and Topic")
            st.write(result["sentiment_summary_chain"])
