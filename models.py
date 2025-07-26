from pydantic import BaseModel, HttpUrl, Field
from typing import List

class QuestionRequest(BaseModel):
    documents: HttpUrl
    questions: List[str] = Field(min_items=1)

class AnswerResponse(BaseModel):
    answers: List[str]
