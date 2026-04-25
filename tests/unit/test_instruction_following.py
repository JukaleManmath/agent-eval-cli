from tests.conftest import make_session, make_test_case

from agenteval.scorers.instruction_following import score_instruction_following


def test_no_forbidden_no_required():
    session = make_session(["Your booking is confirmed"])
    tc = make_test_case(forbidden_phrases=[], required_phrases=[])
    result = score_instruction_following(session, tc)
    assert result.score == 1.0


def test_forbidden_phrase_present():
    session = make_session(["I cannot help you with that"])
    tc = make_test_case(forbidden_phrases=["I cannot help"])
    result = score_instruction_following(session, tc)
    assert result.score < 1.0
    assert any("I cannot help" in evidence for evidence in result.evidence)


def test_forbidden_phrase_absent():
    session = make_session(["Your booking is confirmed"])
    tc = make_test_case(forbidden_phrases=["I cannot help"])
    result = score_instruction_following(session, tc)
    assert result.score == 1.0


def test_required_phrase_present():
    session = make_session(["Your booking reference is AB123"])
    tc = make_test_case(required_phrases=["booking reference"])
    result = score_instruction_following(session, tc)
    assert result.score == 1.0


def test_required_phrase_absent():
    session = make_session(["Thank you for calling"])
    tc = make_test_case(required_phrases=["booking reference"])
    result = score_instruction_following(session, tc)
    assert result.score == 0.5
