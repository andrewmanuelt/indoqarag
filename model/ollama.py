from typing import Annotated, List

from pydantic import BaseModel, Field, RootModel


class ContextJudgment(BaseModel):
    context_index: int
    verdict: int = Field(ge=0, le=1)
    reason: str


class PrecisionOutput(RootModel[dict[str, Annotated[int, Field(strict=True, ge=0, le=1)]]]):
    pass


class GradedContextJudgment(BaseModel):
    context_index: int
    relevance: int = Field(ge=0, le=4)
    reason: str


class RelevanceOutput(BaseModel):
    judgments: List[GradedContextJudgment]


class AtomicStatements(BaseModel):
    statements: List[str]


class StatementJudgment(BaseModel):
    attributed: int = Field(ge=0, le=1)
    reason: str


class RecallOutput(RootModel[dict[str, StatementJudgment]]):
    pass