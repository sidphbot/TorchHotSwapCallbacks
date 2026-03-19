"""Convert confirmed hypotheses to recipe entries."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .types import HypothesisNode


def hypothesis_to_recipe_entry(hyp: HypothesisNode) -> dict:
    """Convert a single confirmed hypothesis to a recipe-compatible command dict."""
    intervention = hyp.intervention
    entry = {
        "module": intervention.get("module", "opt"),
        "op": intervention.get("op", "set_params"),
        "step": hyp.step_created,
        "source": f"research:{hyp.node_id}",
    }
    if "params" in intervention:
        entry["params"] = intervention["params"]
    if "target" in intervention:
        entry["target"] = intervention["target"]
    if "id" in intervention:
        entry["id"] = intervention["id"]
    return entry


def export_confirmed_recipe(
    hypotheses: List[HypothesisNode],
    out_path: str,
) -> str:
    """Export confirmed hypotheses as a recipe JSONL file."""
    confirmed = [h for h in hypotheses if h.status in ("confirmed", "exported")]
    confirmed.sort(key=lambda h: h.step_created)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for hyp in confirmed:
            entry = hypothesis_to_recipe_entry(hyp)
            f.write(json.dumps(entry) + "\n")
    return out_path
