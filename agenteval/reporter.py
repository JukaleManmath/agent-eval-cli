from __future__ import annotations

import uuid
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from agenteval.schema.report import (
    ConversationTurn,
    Report,
    RunConfig,
    RunSummary,
    ScenarioReport,
    ScoreDetail,
)


def write_report(
    results: list,
    summary: RunSummary,
    mode: str,
    model: str | None,
    output_dir: Path,
) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summary = _ensure_summary(summary)
    scenarios = [_to_scenario_report(result, mode) for result in results]
    report = Report(
        agenteval_version=_get_version(),
        run_id=str(uuid.uuid4()),
        timestamp=timestamp,
        simulator_mode=mode,
        simulator_model=model,
        run_config=RunConfig(mode=mode, concurrency=1),
        summary=run_summary,
        scenarios=scenarios,
    )

    output_path = output_dir / f"agenteval_report_{timestamp}.json"
    output_path.write_text(report.model_dump_json(indent=2))
    return output_path


def _ensure_summary(summary: RunSummary | dict[str, Any]) -> RunSummary:
    if isinstance(summary, RunSummary):
        return summary
    return RunSummary(**summary)


def _to_scenario_report(result: Any, mode: str) -> ScenarioReport:
    if hasattr(result, "to_report_dict"):
        data = result.to_report_dict()
        if not data.get("simulator_mode"):
            data["simulator_mode"] = mode
        return ScenarioReport(**data)

    scores = {
        name: ScoreDetail(
            score=scorer_result.score,
            passed=scorer_result.passed,
            evidence=scorer_result.evidence,
        )
        for name, scorer_result in getattr(result, "scores", {}).items()
    }
    conversation = [
        ConversationTurn(number=turn.number, role=turn.role, content=turn.content)
        for turn in getattr(result, "conversation", [])
    ]

    return ScenarioReport(
        scenario_id=result.scenario_id,
        scenario_name=result.scenario_name,
        outcome_type=result.outcome_type,
        session_id=result.session_id,
        simulator_mode=getattr(result, "simulator_mode", mode),
        passed=result.passed,
        aggregate_score=result.aggregate_score,
        aggregate_weights_used=getattr(result, "aggregate_weights_used", {}),
        turns_used=result.turns_used,
        termination_reason=result.termination_reason,
        scores=scores,
        conversation=conversation,
        errored=getattr(result, "errored", False),
        error=getattr(result, "error", None),
        error_type=getattr(result, "error_type", None),
    )


def _get_version() -> str:
    try:
        return version("agent-eval-cli")
    except PackageNotFoundError:
        return "0.1.0"
