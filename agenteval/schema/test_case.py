from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    endpoint: str
    method: str = "POST"
    headers: dict[str, str] = Field(default_factory=dict)
    request_template: str
    response_path: str
    timeout_seconds: int = 30
    health_check_path: str | None = None


class UserPersona(BaseModel):
    name: str
    tone: str
    background: str
    opening_message: str


class ConversationConfig(BaseModel):
    max_turns: int
    min_turns: int
    expected_turns: int


class Evaluation(BaseModel):
    success_intent: str | None = None
    success_keywords: list[str] = Field(default_factory=list)
    refusal_intent: str | None = None
    refusal_keywords: list[str] = Field(default_factory=list)
    escalation_intent: str | None = None
    escalation_keywords: list[str] = Field(default_factory=list)
    forbidden_phrases: list[str] = Field(default_factory=list)
    required_phrases: list[str] = Field(default_factory=list)
    context_facts: list[str] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)


class Meta(BaseModel):
    schema_version: str = "1.0"
    scenario_id: str
    name: str
    description: str | None = None
    version: str | None = None
    tags: list[str] = Field(default_factory=list)


class TestCase(BaseModel):
    meta: Meta
    agent: AgentConfig
    user_persona: UserPersona
    outcome_type: Literal["success", "refusal", "escalation"]
    policy_reason: str | None = None
    conversation: ConversationConfig
    evaluation: Evaluation
    scripted_turns: list[str] = Field(default_factory=list)


def load_test_case(path: Path) -> TestCase:
    if not path.exists():
        raise FileNotFoundError(f"Test case file not found: {path}")
    data = yaml.safe_load(path.read_text())
    return TestCase(**data)
