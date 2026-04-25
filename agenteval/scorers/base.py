from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScorerResult:
    score: float | None
    passed: bool | None
    evidence: list[str] = field(default_factory=list)


def make_result(
    score: float | None,
    threshold: float,
    evidence: list[str] | None = None,
) -> ScorerResult:
    passed = None if score is None else score >= threshold
    return ScorerResult(
        score=score,
        passed=passed,
        evidence=evidence or [],
    )
