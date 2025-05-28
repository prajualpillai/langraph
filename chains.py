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
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

# Initializing chains
llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="llama3-8b-8192") # distill-whisper-large-v3, deepseek-r1-distill-llama-70b

generate_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm


