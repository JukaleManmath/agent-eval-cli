from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScoreDetail(BaseModel):
    score: float | None
    passed: bool | None
    evidence: list[str] = Field(default_factory=list)


class ConversationTurn(BaseModel):
    number: int
    role: Literal["user", "agent"]
    content: str


class ScenarioReport(BaseModel):
    scenario_id: str
    scenario_name: str
    outcome_type: str
    session_id: str
    simulator_mode: str
    passed: bool
    aggregate_score: float
    aggregate_weights_used: dict[str, float] = Field(default_factory=dict)
    turns_used: int
    termination_reason: str
    scores: dict[str, ScoreDetail] = Field(default_factory=dict)
    conversation: list[ConversationTurn] = Field(default_factory=list)
    errored: bool = False
    error: str | None = None
    error_type: str | None = None


class RunSummary(BaseModel):
    total_scenarios: int
    passed: int
    failed: int
    errored: int
    aggregate_score: float
    aggregate_score_method: str = "plain_mean"
    min_scenario_score: float
    overall_pass: bool
    aggregate_note: str = (
        "aggregate_score is the plain mean of all non-errored scenario scores. "
        "A single failing scenario can be masked by high scores elsewhere - "
        "always check min_scenario_score for outliers."
    )


class RunConfig(BaseModel):
    mode: str
    concurrency: int
    fail_on_threshold: float | None = None
    scorers_active: list[str] = Field(default_factory=list)
    scorers_skipped: list[str] = Field(default_factory=list)
    skip_reason: str | None = None


class Report(BaseModel):
    schema_version: str = "1.0"
    agenteval_version: str
    run_id: str
    timestamp: str
    simulator_mode: str
    simulator_model: str | None = None
    run_config: RunConfig
    summary: RunSummary
    scenarios: list[ScenarioReport] = Field(default_factory=list)
