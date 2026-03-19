"""ResearchEngine — central orchestrator for the research & learning module."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .graph import ResearchGraph
from .types import HypothesisNode, EvidenceNode, ObservationNode
from .recipe_gen import hypothesis_to_recipe_entry, export_confirmed_recipe


class ResearchEngine:
    """Orchestrates research graph, wires to autopilot and effect tracker."""

    def __init__(self, run_dir: str, nn_mode: bool = False):
        self._run_dir = run_dir
        self._graph = ResearchGraph(run_dir)
        self._effect_tracker = None
        self._autopilot = None
        self._nn_mode = nn_mode
        self._learner = None  # lazy init
        self._seen_effect_ids: set = set()
        self._seen_mutation_refs: set = set()
        self._commands_path = os.path.join(run_dir, "hotcb.commands.jsonl")

        # Try to load existing state
        if not self._graph.load_snapshot():
            self._graph.replay_events()

    @property
    def graph(self) -> ResearchGraph:
        return self._graph

    # ------------------------------------------------------------------
    # Wiring (called from app.py lifespan)
    # ------------------------------------------------------------------

    def set_effect_tracker(self, tracker):
        self._effect_tracker = tracker

    def set_autopilot(self, engine):
        self._autopilot = engine

    # ------------------------------------------------------------------
    # Integration callbacks
    # ------------------------------------------------------------------

    def on_rule_fired(self, action, step: int):
        """Auto-create 'discovered' hypothesis from autopilot rule firing."""
        rule_id = getattr(action, "rule_id", "") or action.get("rule_id", "") if isinstance(action, dict) else getattr(action, "rule_id", "")
        proposed = action if isinstance(action, dict) else getattr(action, "proposed_action", {})
        condition_met = action.get("condition_met", "") if isinstance(action, dict) else getattr(action, "condition_met", "")

        # Dedup by (condition, intervention params)
        params_key = json.dumps(proposed, sort_keys=True) if proposed else ""
        for hyp in self._graph.all_hypotheses():
            existing_key = json.dumps(hyp.intervention, sort_keys=True)
            if hyp.condition == condition_met and existing_key == params_key:
                return hyp  # already exists

        return self._graph.add_hypothesis(
            condition=condition_met,
            intervention=proposed,
            expected_outcome=f"Rule {rule_id} expected improvement",
            step=step,
            source=f"auto_rule:{rule_id}",
            status="discovered",
        )

    def on_effect_completed(self, effect, step: int):
        """Create evidence from a completed effect, update hypothesis confidence."""
        effect_id = f"{effect.param_key}@{effect.step}"
        if effect_id in self._seen_effect_ids:
            return
        self._seen_effect_ids.add(effect_id)

        # Find matching hypotheses by intervention params
        matching = self._find_matching_hypotheses(effect)
        for hyp in matching:
            ev = self._graph.add_evidence(
                hypothesis_id=hyp.node_id,
                outcome=effect.outcome,
                delta=dict(effect.delta) if hasattr(effect, "delta") else {},
                context={
                    "param_key": effect.param_key,
                    "old_value": _safe_val(effect.old_value),
                    "new_value": _safe_val(effect.new_value),
                },
                effect_id=effect_id,
                step=step,
            )
            self._update_confidence(hyp)

        # NN active learning
        if self._nn_mode and self._learner:
            self._learner_on_effect(effect)

    def on_metrics(self, step: int, metrics: dict):
        """Poll effect tracker for new completions."""
        if self._effect_tracker is None:
            return
        for effect in self._effect_tracker.get_completed():
            self.on_effect_completed(effect, step)

    def on_mutation_applied(self, step: int, record: dict):
        """Shadow-track an applied mutation. Creates shadow hypothesis if untracked."""
        # Build a mutation reference key
        param_key = record.get("param_key", "") or record.get("key", "")
        op = record.get("op", "")
        mutation_ref = f"{param_key}:{op}@{step}"
        if mutation_ref in self._seen_mutation_refs:
            return
        self._seen_mutation_refs.add(mutation_ref)

        # Check if any existing hypothesis already covers this mutation
        intervention = {
            "module": record.get("module", ""),
            "op": op,
            "params": record.get("params", {}),
        }
        if self._has_matching_hypothesis(param_key, intervention, step):
            return  # already tracked — no shadow needed

        # Create shadow hypothesis
        self._graph.add_shadow_hypothesis(
            mutation_ref=mutation_ref,
            intervention=intervention,
            step=step,
            source=f"shadow:{record.get('source', 'unknown')}",
        )

    def _has_matching_hypothesis(self, param_key: str, intervention: dict, step: int) -> bool:
        """Check if any non-shadow hypothesis covers this mutation."""
        for hyp in self._graph.all_hypotheses():
            if hyp.status in ("archived", "shadow"):
                continue
            # Match by param_key in intervention params
            params = hyp.intervention.get("params", {})
            if param_key and param_key in params:
                return True
            # Match by module + op
            if (hyp.intervention.get("module") == intervention.get("module")
                    and hyp.intervention.get("op") == intervention.get("op")
                    and hyp.step_created <= step):
                if params and intervention.get("params"):
                    # At least one overlapping param key
                    if set(params) & set(intervention["params"]):
                        return True
        return False

    def acknowledge_shadow(self, hyp_id: str) -> bool:
        """Acknowledge a shadow hypothesis (user reviewed the mutation)."""
        return self._graph.acknowledge_shadow(hyp_id)

    def promote_shadow(self, hyp_id: str, condition: str = "",
                       expected_outcome: str = "") -> bool:
        """Promote shadow to proposed with optional structured info."""
        return self._graph.promote_shadow(hyp_id, condition, expected_outcome)

    @property
    def contamination_status(self) -> dict:
        """Return current contamination status summary."""
        shadows = self._graph.shadow_hypotheses()
        diluted = self._graph.diluted_nodes()
        return {
            "has_contamination": self._graph.has_unacknowledged_shadows,
            "shadow_count": len(shadows),
            "diluted_count": len(diluted),
            "shadow_steps": list(self._graph._shadow_steps),
            "shadow_ids": [s.node_id for s in shadows],
        }

    # ------------------------------------------------------------------
    # Manual operations (CLI/API)
    # ------------------------------------------------------------------

    def observe(
        self,
        text: str,
        step: int = 0,
        tags: Optional[List[str]] = None,
        stream_id: Optional[str] = None,
    ) -> ObservationNode:
        return self._graph.add_observation(
            text=text, step=step, tags=tags, stream_id=stream_id,
        )

    def hypothesize(
        self,
        condition: str,
        intervention: Dict[str, Any],
        expected_outcome: str,
        stream_id: Optional[str] = None,
    ) -> HypothesisNode:
        return self._graph.add_hypothesis(
            condition=condition,
            intervention=intervention,
            expected_outcome=expected_outcome,
            stream_id=stream_id,
        )

    def refine(
        self,
        observation_id: str,
        condition: str,
        intervention: Dict[str, Any],
        expected_outcome: str,
    ) -> HypothesisNode:
        """Create hypothesis from observation, link with refines_to edge."""
        obs = self._graph.get_node(observation_id)
        if obs is None:
            raise ValueError(f"Observation not found: {observation_id}")
        stream_id = getattr(obs, "stream_id", None)
        hyp = self._graph.add_hypothesis(
            condition=condition,
            intervention=intervention,
            expected_outcome=expected_outcome,
            stream_id=stream_id,
        )
        self._graph.add_edge(observation_id, hyp.node_id, "refines_to")
        return hyp

    def test_hypothesis(self, hyp_id: str) -> dict:
        """Write intervention to commands.jsonl, transition to testing."""
        hyp = self._graph.get_node(hyp_id)
        if not isinstance(hyp, HypothesisNode):
            raise ValueError(f"Hypothesis not found: {hyp_id}")
        if hyp.status not in ("proposed",):
            raise ValueError(f"Cannot test hypothesis in status: {hyp.status}")

        # Transition to testing
        self._graph.transition_hypothesis(hyp_id, "testing")
        hyp.test_count += 1

        # Write command
        cmd = hypothesis_to_recipe_entry(hyp)
        cmd["source"] = f"research_test:{hyp_id}"
        os.makedirs(self._run_dir, exist_ok=True)
        with open(self._commands_path, "a") as f:
            f.write(json.dumps(cmd) + "\n")

        return {"hypothesis_id": hyp_id, "command": cmd, "status": "testing"}

    def conclude_hypothesis(self, hyp_id: str, status: str, reason: str = "") -> bool:
        ok = self._graph.transition_hypothesis(hyp_id, status)
        if ok and reason:
            hyp = self._graph.get_node(hyp_id)
            if isinstance(hyp, HypothesisNode):
                hyp.expected_outcome = f"{hyp.expected_outcome} [concluded: {reason}]"
        return ok

    # ------------------------------------------------------------------
    # NN mode (staged)
    # ------------------------------------------------------------------

    def predict_outcome(self, intervention: dict, context: dict) -> float:
        """Tier 1: predict P(improved) for a proposed intervention."""
        learner = self._get_learner()
        if learner is None:
            return 0.5  # no model available
        return learner.predict(intervention, context)

    def suggest_interventions(self, context: dict, top_k: int = 3) -> List[dict]:
        """Tier 2: suggest optimal interventions."""
        learner = self._get_learner()
        if learner is None:
            return []
        return learner.suggest(context, top_k=top_k)

    def discover_patterns(self) -> List[HypothesisNode]:
        """Tier 3: discover patterns from collected data."""
        learner = self._get_learner()
        if learner is None:
            return []
        patterns = learner.discover()
        created = []
        for pat in patterns:
            hyp = self._graph.add_hypothesis(
                condition=pat.get("condition", "discovered pattern"),
                intervention=pat.get("intervention", {}),
                expected_outcome=pat.get("expected_outcome", "improvement"),
                source="nn_discovery",
                status="discovered",
            )
            created.append(hyp)
        return created

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_confirmed_recipe(self, out_path: Optional[str] = None) -> str:
        if out_path is None:
            out_path = os.path.join(self._run_dir, "hotcb.research.recipe.jsonl")
        confirmed = [
            h for h in self._graph.all_hypotheses()
            if h.status in ("confirmed", "exported")
        ]
        return export_confirmed_recipe(confirmed, out_path)

    # ------------------------------------------------------------------
    # Continuation bridge
    # ------------------------------------------------------------------

    def load_continuation_conditions(self, task: str, base_run_dir: str = "") -> list:
        """Import continuation conditions for a task as proposed hypotheses.

        Creates one 'proposed' hypothesis per condition, deduplicating by
        (condition text, intervention params).
        """
        from hotcb.eval.conditions import (
            MNIST_CONTINUATION_CONDITIONS, MNIST_NEW_CONTINUATION_CONDITIONS,
            CIFAR10_CONTINUATION_CONDITIONS, CIFAR10_NEW_CONTINUATION_CONDITIONS,
        )

        task_conditions = {
            "mnist": MNIST_CONTINUATION_CONDITIONS + MNIST_NEW_CONTINUATION_CONDITIONS,
            "cifar10": CIFAR10_CONTINUATION_CONDITIONS + CIFAR10_NEW_CONTINUATION_CONDITIONS,
        }
        conditions = task_conditions.get(task, [])
        if not conditions:
            return []

        created = []
        for cond in conditions:
            # Build intervention dict from recipe_ops
            intervention = {
                "module": "continuation",
                "op": "branch",
                "params": {},
                "recipe_name": cond.recipe_name,
                "resume_mode": cond.resume_mode,
                "task": cond.task,
            }
            for op in cond.recipe_ops:
                intervention["params"][op["target"]] = op.get("value")

            # Dedup by condition + intervention
            condition_text = cond.hypothesis_condition or cond.description
            import json as _json
            params_key = _json.dumps(intervention["params"], sort_keys=True)
            duplicate = False
            for hyp in self._graph.all_hypotheses():
                existing_key = _json.dumps(
                    hyp.intervention.get("params", {}), sort_keys=True
                )
                if hyp.condition == condition_text and existing_key == params_key:
                    duplicate = True
                    break

            if duplicate:
                continue

            hyp = self._graph.add_hypothesis(
                condition=condition_text,
                intervention=intervention,
                expected_outcome=cond.hypothesis_expected or cond.description,
                source=f"continuation:{cond.name}",
                status="proposed",
            )
            created.append(hyp)

        return created

    def launch_condition_tests(
        self, hyp_ids: list, base_run_dir: str
    ) -> list:
        """Launch continuation branches for selected hypotheses.

        For each hypothesis:
          1. Transition to 'testing'
          2. Build ContinuationConfig from intervention dict
          3. Run branch
          4. Create evidence from result
          5. Transition to confirmed/refuted/inconclusive

        Returns list of result dicts.
        """
        from hotcb.routines.continuation import (
            ContinuationConfig, ObjectiveSpec, BudgetSpec,
            MutationRecipe, MutationOp, AnchorSpec,
            ContinuationPlanner, ContinuationLauncher, ResumeMode,
        )
        from hotcb.routines.continuation.anchors import AnchorSelector

        results = []
        for hyp_id in hyp_ids:
            hyp = self._graph.get_node(hyp_id)
            if not isinstance(hyp, HypothesisNode):
                results.append({"hypothesis_id": hyp_id, "error": "not found"})
                continue
            if hyp.status not in ("proposed",):
                results.append({"hypothesis_id": hyp_id, "error": f"cannot test in status {hyp.status}"})
                continue

            # Transition to testing
            self._graph.transition_hypothesis(hyp_id, "testing")

            # Extract config from intervention
            intervention = hyp.intervention
            task = intervention.get("task", "mnist")
            recipe_name = intervention.get("recipe_name", "custom")
            resume_mode_str = intervention.get("resume_mode", "full_resume")
            params = intervention.get("params", {})

            # Build recipe
            ops = []
            for target, value in params.items():
                relative = isinstance(value, (int, float)) and value < 1.0 and target in ("lr", "weight_decay")
                ops.append(MutationOp(target=target, value=value, relative=relative))
            recipe = MutationRecipe(name=recipe_name, ops=ops)

            # Read base metrics
            import json as _json
            base_best = {}
            best_path = os.path.join(base_run_dir, "best_metrics.json")
            if os.path.exists(best_path):
                with open(best_path) as f:
                    base_best = _json.load(f)

            base_final = {}
            metrics_path = os.path.join(base_run_dir, "hotcb.metrics.jsonl")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    lines = f.readlines()
                for line in reversed(lines):
                    line = line.strip()
                    if line:
                        try:
                            rec = _json.loads(line)
                            base_final = rec.get("metrics", rec)
                            break
                        except _json.JSONDecodeError:
                            continue

            # Select anchors
            try:
                selector = AnchorSelector(base_run_dir)
                anchors = selector.select_auto(max_anchors=1)
            except Exception as exc:
                self._graph.transition_hypothesis(hyp_id, "inconclusive")
                results.append({"hypothesis_id": hyp_id, "error": str(exc)})
                continue

            if not anchors:
                self._graph.transition_hypothesis(hyp_id, "inconclusive")
                results.append({"hypothesis_id": hyp_id, "error": "no checkpoints found"})
                continue

            config = ContinuationConfig(
                base_run_dir=base_run_dir,
                base_final_metrics=base_final,
                base_best_metrics=base_best,
                task=task,
                objective=ObjectiveSpec(primary_metric="val_accuracy", mode="max"),
                budget=BudgetSpec(max_extra_steps=500),
                anchors=anchors,
                recipes=[recipe],
                resume_modes=[ResumeMode(resume_mode_str)],
                output_dir=os.path.join(base_run_dir, "continuation", "research"),
            )

            planner = ContinuationPlanner(config)
            branches = planner.plan()
            if not branches:
                self._graph.transition_hypothesis(hyp_id, "inconclusive")
                results.append({"hypothesis_id": hyp_id, "error": "no branches planned"})
                continue

            launcher = ContinuationLauncher(config)
            branch_result = launcher.run_branch(branches[0])

            # Create evidence
            outcome = "neutral"
            if branch_result.status.value == "won":
                outcome = "improved"
            elif branch_result.status.value in ("lost", "unstable", "rejected_guardrail"):
                outcome = "degraded"

            self._graph.add_evidence(
                hypothesis_id=hyp_id,
                outcome=outcome,
                delta={"primary_delta_pct": branch_result.primary_delta_pct},
                context={
                    "branch_id": branch_result.branch_id,
                    "final_metrics": branch_result.final_metrics,
                    "best_metrics": branch_result.best_metrics,
                    "status": branch_result.status.value,
                },
                step=branch_result.total_steps,
            )

            # Conclude
            if outcome == "improved":
                self._graph.transition_hypothesis(hyp_id, "confirmed")
            elif outcome == "degraded":
                self._graph.transition_hypothesis(hyp_id, "refuted")
            else:
                self._graph.transition_hypothesis(hyp_id, "inconclusive")

            results.append({
                "hypothesis_id": hyp_id,
                "outcome": outcome,
                "branch_status": branch_result.status.value,
                "primary_delta_pct": branch_result.primary_delta_pct,
                "run_dir": branch_result.run_dir,
            })

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_matching_hypotheses(self, effect) -> List[HypothesisNode]:
        """Find hypotheses whose intervention matches the effect's param_key."""
        matches = []
        for hyp in self._graph.all_hypotheses():
            if hyp.status in ("archived", "exported"):
                continue
            params = hyp.intervention.get("params", {})
            # Match if param_key appears in intervention params
            if effect.param_key in params:
                matches.append(hyp)
            # Also match by module+op pattern
            if hyp.intervention.get("module") and effect.param_key.startswith(
                hyp.intervention.get("module", "")
            ):
                if hyp not in matches:
                    matches.append(hyp)
        return matches

    def _update_confidence(self, hyp: HypothesisNode):
        """Update hypothesis confidence from evidence ratio."""
        evidence = self._graph.evidence_for(hyp.node_id)
        if not evidence:
            hyp.confidence = 0.0
            return
        supports = sum(1 for e in evidence if e.outcome == "improved")
        contradicts = sum(1 for e in evidence if e.outcome == "degraded")
        total = supports + contradicts
        hyp.confidence = supports / total if total > 0 else 0.5

    def _get_learner(self):
        if not self._nn_mode:
            return None
        if self._learner is None:
            try:
                from .learner.model import OutcomePredictor
                self._learner = OutcomePredictor()
                self._learner.load_pretrained()
            except Exception:
                return None
        return self._learner

    def _learner_on_effect(self, effect):
        try:
            from .learner.features import extract_features
            from .learner.data import TrainingDataset
            features = extract_features(effect)
            dataset_path = os.path.join(self._run_dir, "hotcb.research.training_data.jsonl")
            dataset = TrainingDataset(dataset_path)
            label = 1.0 if effect.outcome == "improved" else 0.0
            dataset.add(features, label)
            if len(dataset) % 10 == 0 and self._learner:
                self._learner.finetune(dataset, epochs=5)
        except Exception:
            pass  # NN is optional, never crash the engine


def _safe_val(v):
    """Make a value JSON-serializable."""
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)
