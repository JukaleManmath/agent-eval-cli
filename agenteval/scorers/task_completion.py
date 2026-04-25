from __future__ import annotations

from agenteval.schema.test_case import TestCase
from agenteval.scorers._embeddings import ML_AVAILABLE, cosine_similarity, embed
from agenteval.scorers.base import ScorerResult, make_result
from agenteval.simulation.session import Session


def score_task_completion(session: Session, test_case: TestCase) -> ScorerResult:
    threshold = test_case.evaluation.thresholds.get("task_completion", 0.70)

    if not ML_AVAILABLE:
        return make_result(score=None, threshold=threshold)

    agent_text = " ".join(turn.content for turn in session.agent_turns())
    criteria = test_case.evaluation.success_intent

    if not agent_text or not criteria:
        return make_result(
            score=None,
            threshold=threshold,
            evidence=["Task completion skipped: missing agent text or goal criteria"],
        )

    agent_embedding = embed(agent_text)
    criteria_embedding = embed(criteria)
    score = cosine_similarity(agent_embedding, criteria_embedding)

    return make_result(
        score=score,
        threshold=threshold,
        evidence=[f"Similarity to success criteria: {score:.3f}"],
    )
