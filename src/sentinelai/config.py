"""Central configuration for the SentinelAI mock."""

from dataclasses import dataclass

# SKAB sensor columns (excluding datetime and labels)
SENSORS: list[str] = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "RateRMS",  # normalized from 'Volume Flow RateRMS' in SKAB CSVs
]

# SKAB raw column name for flow rate (semicolon-separated files)
SKAB_FLOW_COL = "Volume Flow RateRMS"

# Default regression target and optional "new sensor" to add later
REGRESSION_TARGET = "Pressure"
NEW_SENSOR = "Accelerometer2RMS"

# Sliding-window parameters (SKAB is ~1 Hz)
WINDOW_LENGTH = 60
WINDOW_STEP = 5

# Policy: abstain when fewer than this fraction of samples are observed
MIN_COVERAGE = 0.70

# Conformal miscoverage level (1 - alpha = target coverage)
CONFORMAL_ALPHA = 0.10

# Gap injection defaults (for demo / stress testing)
GAP_PROB_PER_SENSOR = 0.0
GAP_MIN_LEN = 3
GAP_MAX_LEN = 15
CORRELATED_GAP_PROB = 0.02
CORRELATED_GAP_LEN = 20

# SKAB scenarios used in notebooks
DEFAULT_SCENARIOS = ("valve1", "valve2")


@dataclass(frozen=True)
class CostMatrix:
    """Asymmetric costs for the decision layer (miss >> false alarm)."""

    cost_fp: float = 1.0  # false alarm / unnecessary escalation
    cost_fn: float = 100.0  # missed hazardous event
    cost_abstain: float = 2.0  # operator burden when uncertain

    def expected_cost(self, p_hazard: float, action: str) -> float:
        """Expected cost of taking `action` given calibrated P(hazard)."""
        p_nom = 1.0 - p_hazard
        if action == "nominal":
            return p_hazard * self.cost_fn
        if action == "alarm":
            return p_nom * self.cost_fp
        if action == "advisory":
            return p_hazard * self.cost_fn * 0.5 + p_nom * self.cost_fp * 0.5
        if action == "abstain":
            return self.cost_abstain + p_hazard * self.cost_fn * 0.3
        raise ValueError(f"Unknown action: {action}")

    @property
    def alarm_vote_p(self) -> float:
        """P(hazard) where nominal and alarm have equal expected cost."""
        return self.cost_fp / (self.cost_fn + self.cost_fp)


@dataclass(frozen=True)
class PolicyConfig:
    """Persistence rules for the policy layer (dwell + hysteresis)."""

    min_coverage: float = MIN_COVERAGE
    # Cost-optimal alarm vote boundary for default CostMatrix (cost_fp / (cost_fn + cost_fp)).
    # vote_threshold: float = 1.0 / 101.0

    # OOD threshold: if more than this fraction of votes are elevated, trigger OOD based on m-of-n persistence.
    ood_threshold: float = 0.5
    # Rolling window over which elevated votes are counted.
    dwell_window: int = 8
    # Advisory gate: arm after this many elevated votes, clear when votes drop below.
    dwell_ood_count: int = 2
    dwell_ood_clear_count: int = 1
    # Advisory gate: arm after this many elevated votes, clear when votes drop below.
    dwell_advise_count: int = 2
    dwell_advise_clear_count: int = 1
    # Alarm gate: needs more persistence to arm, wider hysteresis to clear.
    dwell_alarm_count: int = 4
    dwell_alarm_clear_count: int = 2
    conformal_alpha: float = CONFORMAL_ALPHA


DEFAULT_COST = CostMatrix()
DEFAULT_POLICY = PolicyConfig()
