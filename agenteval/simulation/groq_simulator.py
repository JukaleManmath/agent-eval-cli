from __future__ import annotations

from agenteval.config import settings
from agenteval.schema.test_case import TestCase, UserPersona
from agenteval.simulation.base_simulator import BaseSimulator


MAX_CONTEXT_TOKENS = 4000
GROQ_MODEL = "llama-3.3-70b-versatile"

GROQ_USER_SIMULATOR_PROMPT = """
You are roleplaying as a real user interacting with an AI agent.

PERSONA:
Name: {persona_name}
Background: {persona_background}
Tone: {persona_tone}

YOUR GOAL:
{goal}

EXPECTED OUTCOME:
{outcome_type_instruction}

STRICT RULES:
1. Stay in character as {persona_name}. Never break character.
2. Keep messages SHORT - 1 to 3 sentences. Real users don't write paragraphs.
3. Pursue your goal naturally across multiple turns.
4. If the agent answers your question, move toward your goal.
5. If the agent says something confusing, ask for clarification in character.
6. If the goal has been achieved, respond naturally to close the conversation.
7. If the agent refuses and that is the expected outcome, accept naturally.
8. If the agent is unhelpful after 3 turns, show mild frustration in character.
9. When the conversation outcome is fully achieved, output on its own line:
   [GOAL_ACHIEVED]

CONVERSATION SO FAR:
{conversation_history}

AGENT'S LAST MESSAGE:
{last_agent_message}

Your response as {persona_name} (short, natural, in character):
"""

OUTCOME_INSTRUCTIONS = {
    "success": "You are trying to complete a task. Keep going until it's done.",
    "refusal": "You are making a request the agent should decline. Accept the refusal.",
    "escalation": "You have a complex issue. You want to speak to a human agent.",
}


def _resolve_goal(test_case: TestCase) -> str:
    outcome_type = test_case.outcome_type
    if outcome_type == "success":
        return test_case.evaluation.success_intent or "Complete the user's request."
    if outcome_type == "refusal":
        return (
            test_case.evaluation.refusal_intent
            or "Request something the agent should decline."
        )
    if outcome_type == "escalation":
        return test_case.evaluation.escalation_intent or "Escalate to a human agent."
    return "Complete the conversation naturally."


class GroqSimulator(BaseSimulator):
    def __init__(self, persona: UserPersona, goal: str, outcome_type: str) -> None:
        self.persona = persona
        self.goal = goal
        self.outcome_type = outcome_type
        self._history: list[dict[str, str]] = []
        self._client: object | None = None

    def _get_client(self) -> object:
        try:
            from groq import Groq
        except ImportError as error:
            raise RuntimeError(
                "groq package not installed. Run: pip install agent-eval-cli[groq]"
            ) from error
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        if self._client is None:
            self._client = Groq(api_key=settings.groq_api_key)
        return self._client

    def get_opening_message(self) -> str:
        opening_message = self.persona.opening_message
        self._history.append({"role": "user", "content": opening_message})
        return opening_message

    def get_next_turn(self, turn_index: int, last_agent_response: str) -> str | None:
        self._ensure_system_prompt(last_agent_response)
        self._history.append({"role": "assistant", "content": last_agent_response})
        messages = self._truncate_history(self._history)

        client = self._get_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=120,
        )
        next_turn = response.choices[0].message.content

        if not next_turn:
            return None

        self._history.append({"role": "user", "content": next_turn})
        return next_turn

    def _ensure_system_prompt(self, last_agent_response: str) -> None:
        prompt = GROQ_USER_SIMULATOR_PROMPT.format(
            persona_name=self.persona.name,
            persona_background=self.persona.background,
            persona_tone=self.persona.tone,
            goal=self.goal,
            outcome_type_instruction=OUTCOME_INSTRUCTIONS.get(
                self.outcome_type,
                "Complete the conversation naturally.",
            ),
            conversation_history=self._format_conversation_history(),
            last_agent_message=last_agent_response,
        )

        if self._history and self._history[0]["role"] == "system":
            self._history[0]["content"] = prompt
        else:
            self._history.insert(0, {"role": "system", "content": prompt})

    def _format_conversation_history(self) -> str:
        if not self._history:
            return "No previous messages."

        lines = []
        for message in self._history:
            if message["role"] == "system":
                continue
            lines.append(f"{message['role']}: {message['content']}")
        return "\n".join(lines) or "No previous messages."

    def _truncate_history(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        total = sum(len(message["content"]) // 4 for message in history)

        while total > MAX_CONTEXT_TOKENS and len(history) > 3:
            removed_user = history.pop(1)
            removed_agent = history.pop(1)
            total -= (
                len(removed_user["content"]) + len(removed_agent["content"])
            ) // 4

        return history
