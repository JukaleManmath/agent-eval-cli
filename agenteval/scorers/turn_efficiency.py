from __future__ import annotations

from agenteval.schema.test_case import TestCase
from agenteval.scorers.base import ScorerResult, make_result
from agenteval.simulation.session import Session


def score_turn_efficiency(session: Session, test_case: TestCase) -> ScorerResult:
    actual = session.turns_used
    expected = test_case.conversation.expected_turns
    min_turns = test_case.conversation.min_turns
    terminated = session.termination_reason
    outcome = test_case.outcome_type
    threshold = test_case.evaluation.thresholds.get("turn_efficiency", 0.60)

    if outcome in ("refusal", "escalation"):
        if terminated != "goal_achieved":
            return make_result(
                score=0.0,
                threshold=threshold,
                evidence=["Outcome never reached"],
            )

        if actual <= expected:
            score = 1.0
            evidence = f"Goal achieved in {actual} turns, expected {expected}"
        else:
            score = max(0.3, 1.0 - ((actual - expected) / expected) * 0.5)
            evidence = (
                f"Completed in {actual} turns, expected {expected} - "
                "agent took too long"
            )

        return make_result(score=score, threshold=threshold, evidence=[evidence])

    if terminated == "max_turns_reached":
        return make_result(
            score=0.0,
            threshold=threshold,
            evidence=["Max turns reached without goal completion"],
        )

    if actual < min_turns:
        evidence = (
            f"Completed in {actual} turns, below min_turns of {min_turns} - "
            "suspiciously fast"
        )
        return make_result(score=0.5, threshold=threshold, evidence=[evidence])

    if actual <= expected:
        score = 1.0
        evidence = f"Goal achieved in {actual} turns, expected {expected}"
    elif actual <= expected * 1.5:
        ratio = (actual - expected) / (expected * 0.5)
        score = max(0.5, 1.0 - ratio * 0.5)
        evidence = (
            f"Completed in {actual} turns, expected {expected} - "
            "agent took too long"
        )
    else:
        score = 0.3
        evidence = (
            f"Completed in {actual} turns, expected {expected} - "
            "agent took too long"
        )

    return make_result(score=score, threshold=threshold, evidence=[evidence])
