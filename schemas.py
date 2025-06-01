# File will hold the schemas for the outputs that we want
from typing import List
from pydantic import BaseModel, Field


class Reflection(BaseModel):

    missing:str = Field(description="Critique of what is missing")
    superfluous:str = Field(description="Critique of what is superflous")


class AnswerQuestion(BaseModel):
    """Answer the question"""
    answer:str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your critique of the answer")
    search_queries: List[str] = Field(description="""1-3 search queries for researching improvements to address the critique of
                                       the current answer""")


class ReviseAnswer(AnswerQuestion):
    """Revise the answer using the critique"""
    refrences:List[str] = Field(description="Citations motivating your updated answer")
    
    
