"""Manipulation risk scoring and cross-modal conflict detection -- PDF Section 6.6.5,
build brief Section 20."""
from emafrs.risk.cross_modal_conflict import ConflictReport, compute_agreement
from emafrs.risk.manipulation_risk import (
    DEFAULT_WEIGHTS,
    ManipulationRiskWeights,
    compute_manipulation_risk,
    flag_manipulation,
    rolling_flag_threshold,
)

__all__ = [
    "ConflictReport", "compute_agreement", "DEFAULT_WEIGHTS", "ManipulationRiskWeights",
    "compute_manipulation_risk", "flag_manipulation", "rolling_flag_threshold",
]
