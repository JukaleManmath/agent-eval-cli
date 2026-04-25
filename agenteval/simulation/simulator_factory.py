from __future__ import annotations

from agenteval.schema.test_case import TestCase
from agenteval.simulation.base_simulator import BaseSimulator
from agenteval.simulation.groq_simulator import GroqSimulator, _resolve_goal
from agenteval.simulation.scripted_simulator import ScriptedSimulator


def create_simulator(mode: str, test_case: TestCase) -> BaseSimulator:
    if mode == "scripted":
        return ScriptedSimulator(
            scripted_turns=test_case.scripted_turns,
            opening_message=test_case.user_persona.opening_message,
        )
    if mode == "groq":
        return GroqSimulator(
            persona=test_case.user_persona,
            goal=_resolve_goal(test_case),
            outcome_type=test_case.outcome_type,
        )
    raise ValueError(f"Unknown simulation mode: '{mode}'. Use 'scripted' or 'groq'.")
