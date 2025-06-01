import datetime
from math import acos
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import (JsonOutputToolsParser, PydanticToolsParser)
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()
from schemas import AnswerQuestion, ReviseAnswer

llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="llama-3.3-70b-versatile")

parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert researcher. 
                Current time: {time}. 
                1. {first_instruction}
                2. Reflect and critique your answer. Be severe to maximize improvement.
                3. Recommend search quesries to research answer and further improve your answer.
            You have to necessarily give the search queries as well."""
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format.")
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())


# Initializing chains
# gen_llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="deepseek-r1-distill-llama-70b") # llama3-8b-8192
# ref_llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="gemma2-9b-it") # distill-whisper-large-v3, deepseek-r1-distill-llama-70b

# generate_chain = generation_prompt | gen_llm
# reflection_chain = reflection_prompt | ref_llm

structured_llm = llm.with_structured_output(AnswerQuestion, include_raw=True)
first_resp_pt = actor_prompt_template.partial(first_instruction="Provide a 250 word detailed answer")

first_responder = first_resp_pt | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")

revise_instructions = """
                            Revise your previous answer using the new information.
                                - You should use the previous critique to add important information to the answer.
                                    -You must use numerical citations in your revised answer to ensure it can be verified.
                            
                                    - Add a 'Reference' section to your answer(which doesn't count towards the word limit). in the form of
                                        - [1] www.example1.com
                                        - [2] www.example2.com
                                - You should use the previous critique to remove superfluous information from your answer and make
                                SURE that it is not more than 250 words.
                            """

revisor = (
            actor_prompt_template.partial(first_instruction=revise_instructions)
            | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")
            )
if __name__=="__main__":

    human_message = HumanMessage(
        content="""Write about the AI powered SOC's problem domain.
        List startups that do that and have raised capital"""
    )

    chain = (
                first_resp_pt 
                | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion") 
                | parser_pydantic
            )
    res = chain.invoke(input={"messages": [human_message]})

    print(res)
