"""
hotcb.scenarios.base — Data classes for the scenario testing system.

Each scenario is a short, reproducible training run that demonstrates a
policy pack rule firing.  Scenarios are structured as isolated external
projects following the INTEGRATION.md Option B pattern.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScenarioConfig:
    """Metadata loaded from a scenario's ``scenario.yaml``."""

    name: str
    pack: str
    description: str = ""
    max_steps: int = 100
    key_metric: str = "train_loss"
    framework: str = "bare"  # "bare" or "lightning"
    expected_rules: List[str] = field(default_factory=list)
    expected_effect: Dict[str, str] = field(default_factory=dict)
    artifact_dir: Optional[str] = None
    seed: int = 42


@dataclass
class ScenarioResult:
    """Outcome of a single scenario run."""

    name: str
    passed: bool
    rules_fired: List[str] = field(default_factory=list)
    expected_rules: List[str] = field(default_factory=list)
    steps_run: int = 0
    error: Optional[str] = None
    applied_commands: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def missing_rules(self) -> List[str]:
        """Rules that were expected but did not fire."""
        return [r for r in self.expected_rules if r not in self.rules_fired]
