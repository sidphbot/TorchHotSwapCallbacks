"""Research & Learning module — structured hypothesis graphs with formal testing."""

from .types import (
    ObservationNode,
    HypothesisNode,
    EvidenceNode,
    ResearchEdge,
    ResearchStream,
    HYP_TRANSITIONS,
)
from .graph import ResearchGraph
from .engine import ResearchEngine

__all__ = [
    "ObservationNode",
    "HypothesisNode",
    "EvidenceNode",
    "ResearchEdge",
    "ResearchStream",
    "HYP_TRANSITIONS",
    "ResearchGraph",
    "ResearchEngine",
]
