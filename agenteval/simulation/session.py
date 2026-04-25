from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Turn:
    number: int
    role: Literal["user", "agent"]
    content: str
    flags: list[str] = field(default_factory=list)


@dataclass
class Session:
    scenario_id: str
    session_id: str
    turns: list[Turn]
    turns_used: int
    termination_reason: str

    def agent_turns(self) -> list[Turn]:
        return [turn for turn in self.turns if turn.role == "agent"]

    def user_turns(self) -> list[Turn]:
        return [turn for turn in self.turns if turn.role == "user"]
