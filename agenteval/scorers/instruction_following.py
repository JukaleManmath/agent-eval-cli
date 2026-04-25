from __future__ import annotations

from agenteval.schema.test_case import TestCase
from agenteval.scorers.base import ScorerResult, make_result
from agenteval.simulation.session import Session


def score_instruction_following(
    session: Session,
    test_case: TestCase,
) -> ScorerResult:
    agent_turns = session.agent_turns()
    total_agent_turns = len(agent_turns)
    violations = 0
    evidence: list[str] = []

    for turn in agent_turns:
        turn_content = turn.content.lower()
        for phrase in test_case.evaluation.forbidden_phrases:
            if phrase.lower() in turn_content:
                violations += 1
                evidence.append(f"Turn {turn.number}: '{phrase}' found")

    if total_agent_turns:
        forbidden_score = max(0.0, 1.0 - (violations / total_agent_turns))
    else:
        forbidden_score = 1.0

    required_phrases = test_case.evaluation.required_phrases
    all_agent_text = " ".join(turn.content for turn in agent_turns).lower()

    if required_phrases:
        required_found = sum(
            1 for phrase in required_phrases if phrase.lower() in all_agent_text
        )
        required_score = required_found / len(required_phrases)
    else:
        required_score = 1.0

    final_score = (0.5 * forbidden_score) + (0.5 * required_score)
    threshold = test_case.evaluation.thresholds.get("instruction_following", 0.80)

    return make_result(
        score=final_score,
        threshold=threshold,
        evidence=evidence,
    )
