"""Load-bearing decision layer: cost rule + dwell + hysteresis + abstain."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from sentinelai.audit.record import AuditRecord, new_audit_record
from sentinelai.config import CostMatrix, DEFAULT_COST, DEFAULT_POLICY, PolicyConfig


@dataclass
class HysteresisGate:
    """Schmitt trigger: arm at a high vote count, clear at a lower one.

    Once armed it stays latched until votes fall below `clear_count`, which is
    what stops an alarm/advisory from chattering near the threshold.
    """

    arm_count: int
    clear_count: int
    _on: bool = False

    def update(self, votes: int) -> bool:
        self._on = votes >= (self.clear_count if self._on else self.arm_count)
        return self._on

    def reset(self) -> None:
        self._on = False


@dataclass
class PolicyLayer:
    """
    Maps calibrated probability + conformal set -> Nominal | Advisory | Alarm | Abstain.

    Never auto-trips on a single window: requires dwell (N-of-M persistence).
    Uses hysteresis: separate clear threshold to drop out of alarm state.
    """

    cost: CostMatrix = field(default_factory=lambda: DEFAULT_COST)
    config: PolicyConfig = field(default_factory=lambda: DEFAULT_POLICY)

    def __post_init__(self) -> None:
        cfg = self.config
        self._alarm_gate = HysteresisGate(cfg.dwell_alarm_count, cfg.dwell_alarm_clear_count)
        self._advise_gate = HysteresisGate(cfg.dwell_advise_count, cfg.dwell_advise_clear_count)
        self._ood_gate = HysteresisGate(cfg.dwell_ood_count, cfg.dwell_ood_clear_count)
        self._recent_alarm_votes: deque[bool] = deque(maxlen=cfg.dwell_window)
        self._recent_ood_votes: deque[bool] = deque(maxlen=cfg.dwell_window)

    def reset(self) -> None:
        self._alarm_gate.reset()
        self._advise_gate.reset()
        self._ood_gate.reset()
        self._recent_alarm_votes.clear()
        self._recent_ood_votes.clear()


    def _cost_optimal_action(self, p: float) -> str:
        candidates = ["nominal", "alarm"]
        return min(candidates, key=lambda a: self.cost.expected_cost(p, a))

    def decide(
        self,
        p_hazard_raw: float,
        p_hazard_calibrated: float,
        ood_score: float=0.0,
        coverage: float=1.0,
    ) -> AuditRecord:
        cfg = self.config
        p = p_hazard_calibrated

        # --- abstain paths ---
        if coverage < cfg.min_coverage:
            return new_audit_record(
                action="abstain",
                p_hazard=p_hazard_raw,
                p_hazard_calibrated=p,
                ood_score=ood_score,
                coverage=coverage,
                dwell_votes=0,
                reason=f"coverage {coverage:.2f} below min {cfg.min_coverage}",
            )
        
        # --- cost-based candidate for this window---
        single_window_candidate = self._cost_optimal_action(p)

        # --- per-window vote + N-of-M dwell persistence ---
        self._recent_alarm_votes.append(single_window_candidate == "alarm")
        dwell_votes = sum(self._recent_alarm_votes)
        self._recent_alarm_votes = deque(
            self._recent_alarm_votes, maxlen=cfg.dwell_window
        )

        # --- two latched hysteresis gates, ranked by severity ---
        alarm_on = self._alarm_gate.update(dwell_votes)
        advise_on = self._advise_gate.update(dwell_votes)

        self._recent_ood_votes.append(ood_score >= cfg.ood_threshold)
        ood_votes = sum(self._recent_ood_votes)
        self._recent_ood_votes = deque(
            self._recent_ood_votes, maxlen=cfg.dwell_window
        )
        ood_on = self._ood_gate.update(ood_votes)

        if ood_on:
            candidate = "abstain"
        elif alarm_on:
            candidate = "alarm"
        elif advise_on:
            candidate = "advisory"
        else:
            candidate = "nominal"

        reason = (
            f"{candidate}: dwell {dwell_votes}/{cfg.dwell_window} "
            f"(alarm arm/clear {cfg.dwell_alarm_count}/{cfg.dwell_alarm_clear_count}, "
            f"advise arm/clear {cfg.dwell_advise_count}/{cfg.dwell_advise_clear_count}, "
            f"ood arm/clear {cfg.dwell_ood_count}/{cfg.dwell_ood_clear_count})"
        )

        return new_audit_record(
            action=candidate,
            p_hazard=p_hazard_raw,
            p_hazard_calibrated=p,
            ood_score=ood_score,
            coverage=coverage,
            dwell_votes=dwell_votes,
            reason=reason,
            expected_cost=self.cost.expected_cost(p, candidate),
        )


def regression_policy_from_interval(
    upper: float,
    lower: float,
    limit: float,
    coverage: float,
    min_coverage: float = DEFAULT_POLICY.min_coverage,
) -> str:
    """
    Simple policy for regression track: escalate on upper bound vs limit.
    Abstain when interval is wide or input coverage is low.
    """
    if coverage < min_coverage:
        return "abstain"
    width = upper - lower
    if width > 0.5 * abs(limit):
        return "abstain"
    if upper > limit:
        return "alarm"
    if upper > 0.8 * limit:
        return "advisory"
    return "nominal"
