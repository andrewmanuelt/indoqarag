from pydantic import BaseModel, Field
from typing import List


class ContextJudgment(BaseModel):
    context_index: int
    verdict: int = Field(ge=0, le=1)
    reason: str


class PrecisionOutput(BaseModel):
    judgments: List[ContextJudgment]


class GradedContextJudgment(BaseModel):
    context_index: int
    relevance: int = Field(ge=0, le=4)
    reason: str


class RelevanceOutput(BaseModel):
    judgments: List[GradedContextJudgment]


class AtomicStatements(BaseModel):
    statements: List[str]


class StatementJudgment(BaseModel):
    statement_index: int
    attributed: int = Field(ge=0, le=1)
    reason: str


class RecallOutput(BaseModel):
    judgments: List[StatementJudgment]