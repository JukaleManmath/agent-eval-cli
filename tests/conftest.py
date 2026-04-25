from agenteval.schema.test_case import (
    AgentConfig,
    ConversationConfig,
    Evaluation,
    Meta,
    TestCase,
    UserPersona,
)
from agenteval.simulation.session import Session, Turn


def make_session(
    agent_texts: list[str],
    user_texts: list[str] | None = None,
    turns_used: int = 2,
    termination: str = "goal_achieved",
    scenario_id: str = "test_scenario",
) -> Session:
    turns = []
    turn_number = 1
    agent_list = agent_texts
    user_list = user_texts or ["hello"] * len(agent_texts)

    for user_text, agent_text in zip(user_list, agent_list):
        turns.append(Turn(number=turn_number, role="user", content=user_text))
        turn_number += 1
        turns.append(Turn(number=turn_number, role="agent", content=agent_text))
        turn_number += 1

    return Session(
        scenario_id=scenario_id,
        session_id="sess-001",
        turns=turns,
        turns_used=turns_used,
        termination_reason=termination,
    )


def make_test_case(
    outcome_type: str = "success",
    max_turns: int = 10,
    min_turns: int = 1,
    expected_turns: int = 4,
    forbidden_phrases: list[str] | None = None,
    required_phrases: list[str] | None = None,
    success_intent: str | None = "agent confirms the booking",
    context_facts: list[str] | None = None,
    thresholds: dict | None = None,
) -> TestCase:
    return TestCase(
        meta=Meta(
            schema_version="1.0",
            scenario_id="test_scenario",
            name="Test Scenario",
        ),
        agent=AgentConfig(
            endpoint="http://localhost:8000/chat",
            request_template='{"message": "${USER_MESSAGE}"}',
            response_path="response",
        ),
        user_persona=UserPersona(
            name="Alice",
            tone="friendly",
            background="customer",
            opening_message="Hello",
        ),
        outcome_type=outcome_type,
        conversation=ConversationConfig(
            max_turns=max_turns,
            min_turns=min_turns,
            expected_turns=expected_turns,
        ),
        evaluation=Evaluation(
            success_intent=success_intent,
            forbidden_phrases=forbidden_phrases or [],
            required_phrases=required_phrases or [],
            context_facts=context_facts or [],
            thresholds=thresholds or {},
        ),
    )
