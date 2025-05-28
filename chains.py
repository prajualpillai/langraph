import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user."
            "Always provide detailed recommendations, including requests for length, virality, style, etc"
            "Do not just say that the tweet is good or bad, give detailed critique"
            "Do not ask for newer tweets, try to improve upon the existing tweet even after the initial improvements are made"
            "Do not give a review of why the changes are impactful"
            "If no further changes are present, just tell the user that the tweet is good enough"
        ),
        MessagesPlaceholder(variable_name="messages")
    ])

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant, tasked with writing excellent twitter posts"
            "Generate the best twitter post possible as per the user's request"
            "If the user provides a critique of the post, consider it and rectify the mistakes made in the previous attempt."
            "Given the critique only give the modified text based on the critique, don't include extra conversations like check it out and other formalities"
            "Always rewrite the review according to the critique given"
            "Do not just say that you'll keep the recommendation in mind for future tweets, re-write the original tweet according to the cumulative critique every time"
            "Do not give ideas on how to improve, simply improve the tweet"
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

# Initializing chains
gen_llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="deepseek-r1-distill-llama-70b") # llama3-8b-8192
ref_llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="gemma2-9b-it") # distill-whisper-large-v3, deepseek-r1-distill-llama-70b

generate_chain = generation_prompt | gen_llm
reflection_chain = reflection_prompt | ref_llm


