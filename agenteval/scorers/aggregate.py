from __future__ import annotations

from agenteval.scorers.base import ScorerResult


BASE_WEIGHTS = {
    "task_completion": 0.30,
    "instruction_following": 0.25,
    "coherence": 0.20,
    "turn_efficiency": 0.15,
    "hallucination_risk": 0.10,
}


def calculate_aggregate(
    scores: dict[str, ScorerResult],
) -> tuple[float, dict[str, float]]:
    active_scores = {
        name: result
        for name, result in scores.items()
        if result.score is not None and name in BASE_WEIGHTS
    }

    if not active_scores:
        return 0.0, {}

    total_weight = sum(BASE_WEIGHTS[name] for name in active_scores)
    weights_used = {
        name: BASE_WEIGHTS[name] / total_weight for name in active_scores
    }
    aggregate_score = sum(
        result.score * weights_used[name]
        for name, result in active_scores.items()
        if result.score is not None
    )

    return round(aggregate_score, 3), weights_used
