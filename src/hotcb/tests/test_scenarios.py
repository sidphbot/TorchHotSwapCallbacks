"""Tests for the scenario testing system.

Each scenario is a short, reproducible training run that demonstrates a
policy pack rule firing.  These tests verify that every registered scenario
passes headless execution with step_delay=0.
"""
import pytest

from hotcb.scenarios import discover, list_scenarios, get, SCENARIO_REGISTRY
from hotcb.scenarios.base import ScenarioConfig, ScenarioResult
from hotcb.scenarios.runner import ScenarioRunner


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------


class TestScenarioDiscovery:
    def test_discover_finds_all_scenarios(self):
        scenarios = discover()
        assert len(scenarios) >= 12

    def test_list_scenarios_returns_configs(self):
        scenarios = list_scenarios()
        assert all(isinstance(s, ScenarioConfig) for s in scenarios)

    def test_get_known_scenario(self):
        cfg = get("stability_nan")
        assert cfg.name == "stability_nan"
        assert cfg.pack == "stability_basics"

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown scenario"):
            get("nonexistent_scenario_xyz")

    def test_all_scenarios_have_required_fields(self):
        for cfg in list_scenarios():
            assert cfg.name
            assert cfg.pack
            assert cfg.max_steps > 0
            assert len(cfg.expected_rules) > 0


# ---------------------------------------------------------------------------
# Parametrized scenario execution
# ---------------------------------------------------------------------------


# Discover all scenarios for parametrization
discover()
_SCENARIO_NAMES = sorted(SCENARIO_REGISTRY.keys())


@pytest.mark.parametrize("scenario_name", _SCENARIO_NAMES)
def test_scenario_passes(scenario_name, tmp_path):
    """Each scenario should pass: expected rules fire during headless run."""
    config = get(scenario_name)
    runner = ScenarioRunner(step_delay=0)
    result = runner.run(config, run_dir=str(tmp_path))

    assert result.passed, (
        f"Scenario {scenario_name} failed: "
        f"expected={result.expected_rules}, "
        f"fired={result.rules_fired}, "
        f"missing={result.missing_rules}, "
        f"error={result.error}"
    )
    assert result.error is None
    assert len(result.rules_fired) > 0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


class TestScenarioResult:
    def test_missing_rules(self):
        result = ScenarioResult(
            name="test",
            passed=False,
            rules_fired=["a", "b"],
            expected_rules=["a", "c"],
        )
        assert result.missing_rules == ["c"]

    def test_no_missing_rules(self):
        result = ScenarioResult(
            name="test",
            passed=True,
            rules_fired=["a", "b"],
            expected_rules=["a"],
        )
        assert result.missing_rules == []
