from pydantic import BaseModel,Field
from typing import List

class Reflection(BaseModel):
    missing:str=Field(description="Critique of what is missing")
    superfuous:str = Field(description="Critique of what is superfluous")

class AnswerQuestion(BaseModel):
    """Answer the question"""

    answer:str = Field(description="~250 word detailed answer to the question.")
    search_queries:List[str] = Field(description="1-3 search queries for researching improvements to address the critique of your answer.")
    reflection:Reflection = Field(description="Your reflection on the initial answer.")