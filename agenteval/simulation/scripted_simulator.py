from __future__ import annotations

from agenteval.simulation.base_simulator import BaseSimulator


class ScriptedSimulator(BaseSimulator):
    def __init__(self, scripted_turns: list[str], opening_message: str) -> None:
        self.scripted_turns = scripted_turns
        self.opening_message = opening_message

    def get_opening_message(self) -> str:
        return self.opening_message

    def get_next_turn(self, turn_index: int, last_agent_response: str) -> str | None:
        if turn_index < len(self.scripted_turns):
            return self.scripted_turns[turn_index]

        if last_agent_response and not last_agent_response.startswith("ERROR"):
            return "[GOAL_ACHIEVED]"

        return None
