from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from agenteval.schema.test_case import TestCase, load_test_case
from agenteval.scorers import (
    calculate_aggregate,
    score_coherence,
    score_hallucination_risk,
    score_instruction_following,
    score_task_completion,
    score_turn_efficiency,
)
from agenteval.scorers.base import ScorerResult
from agenteval.simulation.engine import run_single_scenario
from agenteval.simulation.session import Session


@dataclass
class ScenarioResult:
    scenario_id: str
    scenario_name: str
    outcome_type: str
    session_id: str
    simulator_mode: str
    passed: bool
    aggregate_score: float
    aggregate_weights_used: dict
    turns_used: int
    termination_reason: str
    scores: dict = field(default_factory=dict)
    conversation: list = field(default_factory=list)
    errored: bool = False
    error: str | None = None
    error_type: str | None = None


@dataclass
class ScenarioError:
    scenario_id: str
    scenario_name: str
    passed: bool = False
    aggregate_score: float = 0.0
    error: str = ""
    error_type: str = ""

    def to_report_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "passed": False,
            "aggregate_score": 0.0,
            "errored": True,
            "error": self.error,
            "error_type": self.error_type,
            "scores": {},
            "conversation": [],
            "turns_used": 0,
            "termination_reason": "agent_error",
            "outcome_type": "",
            "session_id": "",
            "simulator_mode": "",
            "aggregate_weights_used": {},
        }


def _score_session(session: Session, test_case: TestCase, mode: str) -> ScenarioResult:
    scores: dict[str, ScorerResult] = {
        "task_completion": score_task_completion(session, test_case),
        "instruction_following": score_instruction_following(session, test_case),
        "coherence": score_coherence(session, test_case),
        "turn_efficiency": score_turn_efficiency(session, test_case),
        "hallucination_risk": score_hallucination_risk(session, test_case),
    }
    aggregate_score, weights_used = calculate_aggregate(scores)
    threshold = test_case.evaluation.thresholds.get("aggregate", 0.70)

    return ScenarioResult(
        scenario_id=test_case.meta.scenario_id,
        scenario_name=test_case.meta.name,
        outcome_type=test_case.outcome_type,
        session_id=session.session_id,
        simulator_mode=mode,
        passed=aggregate_score >= threshold,
        aggregate_score=aggregate_score,
        aggregate_weights_used=weights_used,
        turns_used=session.turns_used,
        termination_reason=session.termination_reason,
        scores=scores,
        conversation=session.turns,
    )


async def run_scenarios(
    test_cases: list[TestCase],
    mode: str,
    concurrency: int = 4,
) -> list[ScenarioResult | ScenarioError]:
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(test_case: TestCase) -> ScenarioResult:
        async with semaphore:
            session = await run_single_scenario(test_case, mode)
            return _score_session(session, test_case, mode)

    results = await asyncio.gather(
        *(run_one(test_case) for test_case in test_cases),
        return_exceptions=True,
    )

    processed: list[ScenarioResult | ScenarioError] = []
    for test_case, result in zip(test_cases, results):
        if isinstance(result, Exception):
            processed.append(
                ScenarioError(
                    scenario_id=test_case.meta.scenario_id,
                    scenario_name=test_case.meta.name,
                    error=str(result),
                    error_type=type(result).__name__,
                )
            )
        else:
            processed.append(result)  # type: ignore[arg-type]

    return processed


def load_test_cases(path: Path, tag: str | None = None) -> list[TestCase]:
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("*.yaml"))

    test_cases: list[TestCase] = []
    for file_path in files:
        try:
            test_case = load_test_case(file_path)
        except Exception:
            continue

        if tag is None or tag in test_case.meta.tags:
            test_cases.append(test_case)

    return test_cases


def build_summary(
    results: list[ScenarioResult | ScenarioError],
    scores: list[float],
    fail_on_threshold: float | None,
) -> dict:
    errored = sum(isinstance(result, ScenarioError) for result in results)
    threshold = fail_on_threshold if fail_on_threshold is not None else 0.0
    passed = sum(score >= threshold for score in scores)
    failed = sum(score < threshold for score in scores)
    aggregate_score = sum(scores) / len(scores) if scores else 0.0
    min_scenario_score = min(scores) if scores else 0.0
    overall_pass = (
        fail_on_threshold is None or aggregate_score >= fail_on_threshold
    ) and failed == 0

    return {
        "total_scenarios": len(results),
        "passed": passed,
        "failed": failed,
        "errored": errored,
        "aggregate_score": aggregate_score,
        "aggregate_score_method": "plain_mean",
        "min_scenario_score": min_scenario_score,
        "overall_pass": overall_pass,
    }
