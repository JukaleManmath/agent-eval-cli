from unittest.mock import patch

from tests.conftest import make_session, make_test_case


def test_task_completion_no_ml():
    session = make_session(["Your booking is confirmed"])
    tc = make_test_case()
    with patch("agenteval.scorers.task_completion.ML_AVAILABLE", False):
        from agenteval.scorers.task_completion import score_task_completion

        result = score_task_completion(session, tc)
    assert result.score is None
    assert result.passed is None


def test_coherence_no_ml():
    session = make_session(["Hello", "Goodbye"])
    tc = make_test_case()
    with patch("agenteval.scorers.coherence.ML_AVAILABLE", False):
        from agenteval.scorers.coherence import score_coherence

        result = score_coherence(session, tc)
    assert result.score is None
    assert result.passed is None


def test_hallucination_no_ml():
    session = make_session(["Refunds take 5 days"])
    tc = make_test_case(context_facts=["Refunds take 5 business days"])
    with patch("agenteval.scorers.hallucination.ML_AVAILABLE", False):
        from agenteval.scorers.hallucination import score_hallucination_risk

        result = score_hallucination_risk(session, tc)
    assert result.score is None
    assert result.passed is None
