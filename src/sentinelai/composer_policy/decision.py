"""Load-bearing decision layer: cost rule + dwell + hysteresis + abstain."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from sentinelai.audit.record import AuditRecord, new_audit_record
from sentinelai.config import CostMatrix, DEFAULT_COST, DEFAULT_POLICY, PolicyConfig


@dataclass
class PolicyLayer:
    """
    Maps calibrated probability + conformal set -> Nominal | Advisory | Alarm | Abstain.

    Never auto-trips on a single window: requires dwell (N-of-M persistence).
    Uses hysteresis: separate clear threshold to drop out of alarm state.
    """

    cost: CostMatrix = field(default_factory=lambda: DEFAULT_COST)
    config: PolicyConfig = field(default_factory=lambda: DEFAULT_POLICY)
    _alarm_state: bool = False
    _recent_alarm_votes: deque[bool] = field(default_factory=lambda: deque(maxlen=5))

    def reset(self) -> None:
        self._alarm_state = False
        self._recent_alarm_votes.clear()

    def _cost_optimal_action(self, p: float) -> str:
        candidates = ["nominal", "advisory", "alarm"]
        return min(candidates, key=lambda a: self.cost.expected_cost(p, a))

    def decide(
        self,
        p_hazard_raw: float,
        p_hazard_calibrated: float,
        conformal_set: list[str],
        coverage: float,
    ) -> AuditRecord:
        cfg = self.config
        p = p_hazard_calibrated

        # --- abstain paths ---
        if coverage < cfg.min_coverage:
            return new_audit_record(
                action="abstain",
                p_hazard=p_hazard_raw,
                p_hazard_calibrated=p,
                conformal_set=conformal_set,
                coverage=coverage,
                dwell_votes=0,
                reason=f"coverage {coverage:.2f} below min {cfg.min_coverage}",
            )

        if set(conformal_set) == {"nominal", "hazard"}:
            return new_audit_record(
                action="abstain",
                p_hazard=p_hazard_raw,
                p_hazard_calibrated=p,
                conformal_set=conformal_set,
                coverage=coverage,
                dwell_votes=0,
                reason="ambiguous conformal set {nominal, hazard}",
            )

        # --- cost-based candidate ---
        candidate = self._cost_optimal_action(p)

        # --- hysteresis: harder to clear than to arm ---
        if self._alarm_state:
            if p < cfg.clear_threshold:
                self._alarm_state = False
                candidate = "nominal"
            else:
                candidate = "alarm"
        else:
            if p >= cfg.alarm_threshold:
                candidate = "alarm"
            elif p >= cfg.advisory_threshold:
                candidate = "advisory"
            else:
                candidate = "nominal"

        # --- dwell: alarm needs N-of-M persistence ---
        alarm_vote = candidate == "alarm"
        self._recent_alarm_votes.append(alarm_vote)
        dwell_votes = sum(self._recent_alarm_votes)
        self._recent_alarm_votes = deque(
            self._recent_alarm_votes, maxlen=cfg.dwell_window
        )

        if candidate == "alarm":
            if dwell_votes < cfg.dwell_count:
                candidate = "advisory"
                reason = (
                    f"alarm suppressed: dwell {dwell_votes}/{cfg.dwell_count} "
                    f"in last {cfg.dwell_window} windows"
                )
            else:
                self._alarm_state = True
                reason = f"alarm armed: p={p:.3f}, dwell={dwell_votes}"
        else:
            reason = f"cost/hysteresis -> {candidate}, p={p:.3f}"

        return new_audit_record(
            action=candidate,
            p_hazard=p_hazard_raw,
            p_hazard_calibrated=p,
            conformal_set=conformal_set,
            coverage=coverage,
            dwell_votes=dwell_votes,
            reason=reason,
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
