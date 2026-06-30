"""Audit trail for every policy decision."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AuditRecord:
    timestamp: str
    action: str
    p_hazard: float
    p_hazard_calibrated: float
    ood_score: float
    coverage: float
    dwell_votes: int
    reason: str
    model_version: str = "cnn_v1"
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_audit_record(
    action: str,
    p_hazard: float,
    p_hazard_calibrated: float,
    ood_score: float,
    coverage: float,
    dwell_votes: int,
    reason: str,
    **extras: Any,
) -> AuditRecord:
    return AuditRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        action=action,
        p_hazard=p_hazard,
        p_hazard_calibrated=p_hazard_calibrated,
        ood_score=ood_score,
        coverage=coverage,
        dwell_votes=dwell_votes,
        reason=reason,
        extras=extras,
    )
