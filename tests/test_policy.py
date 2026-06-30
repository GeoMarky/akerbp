"""Tests for the policy / decision layer."""

from sentinelai.config import DEFAULT_COST
from sentinelai.policy.decision import (
    HysteresisGate,
    PolicyLayer,
    regression_policy_from_interval,
)

# Cost-optimal vote: alarm wins when p > alarm_vote_p (~0.01 with default costs).
P_BELOW_COST_VOTE = DEFAULT_COST.alarm_vote_p * 0.5
P_SPIKE = 0.9
P_PERSISTENT = 0.95


def test_hysteresis_gate_arms_and_latches():
    gate = HysteresisGate(arm_count=4, clear_count=2)
    assert gate.update(3) is False  # below arm
    assert gate.update(4) is True  # arms
    assert gate.update(3) is True  # latched: above clear, below arm
    assert gate.update(2) is True  # still at clear boundary
    assert gate.update(1) is False  # drops below clear -> off
    assert gate.update(3) is False  # needs arm count again to re-arm


def test_abstain_on_low_coverage():
    policy = PolicyLayer()
    rec = policy.decide(0.9, 0.9, ["hazard"], coverage=0.5)
    assert rec.action == "abstain"
    assert "coverage" in rec.reason


def test_abstain_on_ambiguous_conformal_set():
    policy = PolicyLayer()
    rec = policy.decide(0.5, 0.5, ["nominal", "hazard"], coverage=0.95)
    assert rec.action == "abstain"


def test_dwell_suppresses_single_spike():
    policy = PolicyLayer()
    policy.reset()
    actions = []
    for p in [P_BELOW_COST_VOTE, P_SPIKE, P_BELOW_COST_VOTE, P_BELOW_COST_VOTE, P_BELOW_COST_VOTE]:
        rec = policy.decide(p, p, ["hazard"], coverage=1.0)
        actions.append(rec.action)
    assert "alarm" not in actions or actions.count("alarm") <= 1


def test_alarm_after_persistent_elevation():
    policy = PolicyLayer()
    policy.reset()
    last_action = "nominal"
    for _ in range(6):
        rec = policy.decide(P_PERSISTENT, P_PERSISTENT, ["hazard"], coverage=1.0)
        last_action = rec.action
    assert last_action == "alarm"


def test_regression_policy_abstain_on_wide_interval():
    action = regression_policy_from_interval(
        upper=10.0, lower=0.0, limit=8.0, coverage=0.9
    )
    assert action == "abstain"
