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
GAP_PROB_PER_SENSOR = 0.05
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
        if action == "advisory":
            return p_hazard * self.cost_fn * 0.5 + p_nom * self.cost_fp * 0.5
        if action == "alarm":
            return p_nom * self.cost_fp
        if action == "abstain":
            return self.cost_abstain + p_hazard * self.cost_fn * 0.3
        raise ValueError(f"Unknown action: {action}")


@dataclass(frozen=True)
class PolicyConfig:
    """Thresholds and persistence rules for the policy layer."""

    min_coverage: float = MIN_COVERAGE
    advisory_threshold: float = 0.15
    alarm_threshold: float = 0.35
    clear_threshold: float = 0.10  # hysteresis: lower bar to clear alarm
    dwell_count: int = 3  # N-of-M: need this many alarm votes in window
    dwell_window: int = 5
    conformal_alpha: float = CONFORMAL_ALPHA


DEFAULT_COST = CostMatrix()
DEFAULT_POLICY = PolicyConfig()
