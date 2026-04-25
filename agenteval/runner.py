from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from agenteval.schema.test_case import TestCase, load_test_case
from agenteval.simulation.engine import run_single_scenario
from agenteval.simulation.session import Session


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


async def run_scenarios(
    test_cases: list[TestCase],
    mode: str,
    concurrency: int = 4,
) -> list[Session | ScenarioError]:
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(test_case: TestCase) -> Session:
        async with semaphore:
            return await run_single_scenario(test_case, mode)

    results = await asyncio.gather(
        *(run_one(test_case) for test_case in test_cases),
        return_exceptions=True,
    )

    processed: list[Session | ScenarioError] = []
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
            processed.append(result)

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
    results: list[Session | ScenarioError],
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
