from __future__ import annotations

from agenteval.schema.test_case import TestCase
from agenteval.scorers._embeddings import (
    ML_AVAILABLE,
    cosine_similarity,
    embed_batch,
)
from agenteval.scorers.base import ScorerResult, make_result
from agenteval.simulation.session import Session


def score_coherence(session: Session, test_case: TestCase) -> ScorerResult:
    threshold = test_case.evaluation.thresholds.get("coherence", 0.65)

    if not ML_AVAILABLE:
        return make_result(score=None, threshold=threshold)

    agent_turns = [turn.content for turn in session.agent_turns()]
    if len(agent_turns) < 2:
        return make_result(
            score=1.0,
            threshold=threshold,
            evidence=["Fewer than 2 agent turns; no contradiction possible"],
        )

    embeddings = embed_batch(agent_turns)
    similarities = [
        cosine_similarity(current, next_item)
        for current, next_item in zip(embeddings, embeddings[1:])
    ]
    score = sum(similarities) / len(similarities)

    return make_result(
        score=score,
        threshold=threshold,
        evidence=[f"Average consecutive agent-turn similarity: {score:.3f}"],
    )
