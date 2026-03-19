"""
hotcb.scenarios — Scenario testing system for policy packs.

Discovers scenario directories under the top-level ``scenarios/`` folder,
loads their ``scenario.yaml`` configs, and provides a registry for running
them programmatically or via CLI.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .base import ScenarioConfig, ScenarioResult

__all__ = [
    "ScenarioConfig",
    "ScenarioResult",
    "discover",
    "get",
    "list_scenarios",
    "SCENARIO_REGISTRY",
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCENARIO_REGISTRY: Dict[str, ScenarioConfig] = {}

_SCENARIOS_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "scenarios"


def _load_config(scenario_dir: Path) -> Optional[ScenarioConfig]:
    """Load a ScenarioConfig from a scenario directory."""
    yaml_path = scenario_dir / "scenario.yaml"
    if not yaml_path.exists():
        return None
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or not isinstance(data, dict):
        return None
    return ScenarioConfig(
        name=data.get("name", scenario_dir.name),
        pack=data["pack"],
        description=data.get("description", ""),
        max_steps=data.get("max_steps", 100),
        key_metric=data.get("key_metric", "train_loss"),
        framework=data.get("framework", "bare"),
        expected_rules=data.get("expected_rules", []),
        expected_effect=data.get("expected_effect", {}),
        artifact_dir=data.get("artifact_dir"),
        seed=data.get("seed", 42),
    )


def discover(root: Optional[Path] = None) -> Dict[str, ScenarioConfig]:
    """Scan for scenario directories and populate the registry."""
    root = root or _SCENARIOS_ROOT
    SCENARIO_REGISTRY.clear()
    if not root.is_dir():
        return SCENARIO_REGISTRY
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        cfg = _load_config(entry)
        if cfg is not None:
            SCENARIO_REGISTRY[cfg.name] = cfg
    return SCENARIO_REGISTRY


def get(name: str, root: Optional[Path] = None) -> ScenarioConfig:
    """Get a scenario config by name. Discovers if registry is empty."""
    if not SCENARIO_REGISTRY:
        discover(root)
    if name not in SCENARIO_REGISTRY:
        raise KeyError(f"Unknown scenario: {name!r}. Available: {list(SCENARIO_REGISTRY)}")
    return SCENARIO_REGISTRY[name]


def list_scenarios(root: Optional[Path] = None) -> List[ScenarioConfig]:
    """Return all discovered scenarios."""
    if not SCENARIO_REGISTRY:
        discover(root)
    return list(SCENARIO_REGISTRY.values())


def scenarios_root() -> Path:
    """Return the path to the top-level scenarios directory."""
    return _SCENARIOS_ROOT
