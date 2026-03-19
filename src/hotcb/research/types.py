"""Data types for the research & learning module."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def _uid() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Status state machine
# ---------------------------------------------------------------------------

VALID_HYP_STATUSES = {
    "proposed",
    "discovered",
    "shadow",
    "testing",
    "confirmed",
    "refuted",
    "inconclusive",
    "exported",
    "archived",
}

# from_status -> set of valid to_statuses
HYP_TRANSITIONS: Dict[str, set] = {
    "proposed": {"testing", "archived"},
    "discovered": {"proposed", "archived"},
    "shadow": {"proposed", "discovered", "archived"},
    "testing": {"confirmed", "refuted", "inconclusive", "archived"},
    "confirmed": {"exported", "archived"},
    "refuted": {"archived"},
    "inconclusive": {"proposed", "archived"},
    "exported": {"archived"},
    "archived": set(),
}


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

@dataclass
class ObservationNode:
    """Informal insight recorded during training."""

    node_id: str = field(default_factory=_uid)
    node_type: str = "observation"
    text: str = ""
    step: int = 0
    source: str = "manual"
    tags: List[str] = field(default_factory=list)
    stream_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ObservationNode":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)


@dataclass
class HypothesisNode:
    """Structured, testable claim about training behaviour."""

    node_id: str = field(default_factory=_uid)
    node_type: str = "hypothesis"
    condition: str = ""
    intervention: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""
    status: str = "proposed"
    confidence: float = 0.0
    nn_confidence: float = 0.0
    evidence_count: int = 0
    test_count: int = 0
    step_created: int = 0
    stream_id: Optional[str] = None
    source: str = "manual"
    mutation_ref: Optional[str] = None  # applied ledger entry ref for shadow hyps
    diluted: bool = False  # True if created while unacknowledged shadows exist
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HypothesisNode":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)


@dataclass
class EvidenceNode:
    """Test result linked to a hypothesis."""

    node_id: str = field(default_factory=_uid)
    node_type: str = "evidence"
    hypothesis_id: str = ""
    effect_id: str = ""
    outcome: str = "neutral"  # improved / degraded / neutral / timeout
    delta: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    source: str = "effect_tracker"
    step: int = 0
    diluted: bool = False  # True if created while unacknowledged shadows exist
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceNode":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

VALID_EDGE_TYPES = {"refines_to", "supports", "contradicts"}


@dataclass
class ResearchEdge:
    """Directed edge between research nodes."""

    edge_id: str = field(default_factory=_uid)
    source_id: str = ""
    target_id: str = ""
    edge_type: str = "supports"  # refines_to | supports | contradicts
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchEdge":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)


# ---------------------------------------------------------------------------
# Research streams
# ---------------------------------------------------------------------------

VALID_STREAM_STATUSES = {"active", "paused", "concluded"}


@dataclass
class ResearchStream:
    """Logical grouping of hypotheses and observations."""

    stream_id: str = field(default_factory=_uid)
    name: str = ""
    description: str = ""
    hypothesis_ids: List[str] = field(default_factory=list)
    observation_ids: List[str] = field(default_factory=list)
    status: str = "active"
    conclusion: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchStream":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)
