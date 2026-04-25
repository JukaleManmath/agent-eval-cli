from __future__ import annotations

from agenteval.schema.test_case import TestCase
from agenteval.scorers._embeddings import (
    ML_AVAILABLE,
    cosine_similarity,
    embed,
    embed_batch,
)
from agenteval.scorers.base import ScorerResult, make_result
from agenteval.simulation.session import Session


def score_hallucination_risk(session: Session, test_case: TestCase) -> ScorerResult:
    threshold = test_case.evaluation.thresholds.get("hallucination_risk", 0.60)

    if not ML_AVAILABLE:
        return make_result(score=None, threshold=threshold)

    context_facts = test_case.evaluation.context_facts
    if not context_facts:
        return make_result(
            score=None,
            threshold=threshold,
            evidence=["No context_facts provided; hallucination check skipped"],
        )

    agent_text = " ".join(turn.content for turn in session.agent_turns())
    if not agent_text:
        return make_result(score=None, threshold=threshold)

    agent_embedding = embed(agent_text)
    fact_embeddings = embed_batch(context_facts)
    similarities = [
        cosine_similarity(agent_embedding, fact_embedding)
        for fact_embedding in fact_embeddings
    ]
    score = sum(similarities) / len(similarities)
    evidence = [
        f"Fact {index} similarity: {similarity:.3f}"
        for index, similarity in enumerate(similarities, start=1)
    ]

    return make_result(score=score, threshold=threshold, evidence=evidence)
