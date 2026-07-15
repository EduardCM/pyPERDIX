from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class ValidationIssue:
    stage: str
    severity: str
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


class ValidationError(RuntimeError):
    def __init__(self, issues: Iterable[ValidationIssue]) -> None:
        self.issues = list(issues)
        super().__init__(format_issue_summary(self.issues))


def add_issue(
    issues: list[ValidationIssue],
    stage: str,
    severity: str,
    code: str,
    message: str,
    context: dict[str, Any] | None = None,
) -> None:
    issues.append(
        ValidationIssue(
            stage=stage,
            severity=severity,
            code=code,
            message=message,
            context=context or {},
        )
    )


def check_count(
    issues: list[ValidationIssue],
    stage: str,
    label: str,
    declared: int,
    values: list[Any],
) -> None:
    actual = len(values)
    if declared != actual:
        add_issue(
            issues,
            stage,
            "high",
            "count.mismatch",
            f"{label}={declared}, len={actual}",
            {"field": label, "declared": declared, "actual": actual},
        )


def check_ref(
    issues: list[ValidationIssue],
    stage: str,
    label: str,
    value: int,
    length: int,
    allow_sentinel: bool = True,
    skip_if_empty: bool = False,
) -> None:
    if skip_if_empty and length == 0:
        return
    if value == -1 and allow_sentinel:
        return
    if value < 0 or value >= length:
        sentinel = " or -1" if allow_sentinel else ""
        add_issue(
            issues,
            stage,
            "high",
            "reference.invalid",
            f"{label}={value}, valid range is 0..{length - 1}{sentinel}",
            {"field": label, "value": value, "length": length},
        )


def check_finite_vector(
    issues: list[ValidationIssue],
    stage: str,
    label: str,
    values: tuple[float, ...],
) -> None:
    for axis, value in enumerate(values):
        if not math.isfinite(float(value)):
            add_issue(
                issues,
                stage,
                "high",
                "vector.non_finite",
                f"{label}[{axis}] is not finite: {value}",
            )


def vector_norm(values: tuple[float, ...]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def format_issue(issue: ValidationIssue) -> str:
    return f"[{issue.severity}] {issue.stage}:{issue.code}: {issue.message}"


def format_issue_summary(issues: list[ValidationIssue]) -> str:
    if not issues:
        return "Pipeline validation failed"
    first = format_issue(issues[0])
    if len(issues) == 1:
        return first
    return f"{first} (+{len(issues) - 1} more)"
