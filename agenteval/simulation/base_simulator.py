from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSimulator(ABC):
    @abstractmethod
    def get_opening_message(self) -> str:
        ...

    @abstractmethod
    def get_next_turn(self, turn_index: int, last_agent_response: str) -> str | None:
        ...
