from agenteval.scorers.turn_efficiency import score_turn_efficiency
from tests.conftest import make_session, make_test_case


def test_within_expected_turns():
    session = make_session(["a", "b", "c"], turns_used=3)
    tc = make_test_case(expected_turns=4, min_turns=1)
    result = score_turn_efficiency(session, tc)
    assert result.score == 1.0
    assert result.passed is True


def test_over_expected_within_1_5x():
    session = make_session(["a"] * 5, turns_used=5)
    tc = make_test_case(expected_turns=4, min_turns=1)
    result = score_turn_efficiency(session, tc)
    assert 0.5 <= result.score < 1.0


def test_over_1_5x_expected():
    session = make_session(["a"] * 7, turns_used=7)
    tc = make_test_case(expected_turns=4, min_turns=1)
    result = score_turn_efficiency(session, tc)
    assert result.score == 0.3


def test_below_min_turns():
    session = make_session(["a"], turns_used=1)
    tc = make_test_case(expected_turns=4, min_turns=3)
    result = score_turn_efficiency(session, tc)
    assert result.score == 0.5


def test_max_turns_reached():
    session = make_session(["a"] * 4, turns_used=4, termination="max_turns_reached")
    tc = make_test_case(expected_turns=4, min_turns=1)
    result = score_turn_efficiency(session, tc)
    assert result.score == 0.0
    assert result.passed is False


def test_refusal_fast_no_penalty():
    session = make_session(["a", "b"], turns_used=2, termination="goal_achieved")
    tc = make_test_case(outcome_type="refusal", expected_turns=2, min_turns=1)
    result = score_turn_efficiency(session, tc)
    assert result.score == 1.0
