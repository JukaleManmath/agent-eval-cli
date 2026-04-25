from agenteval.scorers.aggregate import calculate_aggregate
from agenteval.scorers.base import ScorerResult


def make_result(score):
    return ScorerResult(score=score, passed=None if score is None else score >= 0.7)


def test_all_scorers_present():
    scores = {
        "task_completion": make_result(0.8),
        "instruction_following": make_result(0.9),
        "coherence": make_result(0.7),
        "turn_efficiency": make_result(1.0),
        "hallucination_risk": make_result(0.6),
    }
    aggregate, weights = calculate_aggregate(scores)
    assert round(sum(weights.values()), 5) == 1.0
    assert 0.0 <= aggregate <= 1.0


def test_one_scorer_none():
    scores = {
        "task_completion": make_result(0.8),
        "instruction_following": make_result(0.9),
        "coherence": make_result(None),
        "turn_efficiency": make_result(1.0),
        "hallucination_risk": make_result(0.6),
    }
    aggregate, weights = calculate_aggregate(scores)
    assert "coherence" not in weights
    assert round(sum(weights.values()), 5) == 1.0


def test_all_scorers_none():
    scores = {
        "task_completion": make_result(None),
        "instruction_following": make_result(None),
        "coherence": make_result(None),
        "turn_efficiency": make_result(None),
        "hallucination_risk": make_result(None),
    }
    aggregate, weights = calculate_aggregate(scores)
    assert aggregate == 0.0
    assert weights == {}
