"""In-memory research graph with JSONL event-log persistence."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .types import (
    ObservationNode,
    HypothesisNode,
    EvidenceNode,
    ResearchEdge,
    ResearchStream,
    HYP_TRANSITIONS,
    VALID_EDGE_TYPES,
    VALID_STREAM_STATUSES,
    _uid,
)

_NODE_CLS = {
    "observation": ObservationNode,
    "hypothesis": HypothesisNode,
    "evidence": EvidenceNode,
}


class ResearchGraph:
    """Holds nodes, edges, and streams in memory. Persists via append-only JSONL."""

    def __init__(self, run_dir: str):
        self._run_dir = run_dir
        self._nodes: Dict[str, Union[ObservationNode, HypothesisNode, EvidenceNode]] = {}
        self._edges: Dict[str, ResearchEdge] = {}
        self._streams: Dict[str, ResearchStream] = {}
        self._event_log_path = os.path.join(run_dir, "hotcb.research.jsonl")
        self._snapshot_path = os.path.join(run_dir, "hotcb.research.json")
        # Shadow/contamination tracking: steps with unacknowledged shadow mutations
        self._shadow_steps: List[int] = []

    # ------------------------------------------------------------------
    # Mutations (each appends to JSONL event log)
    # ------------------------------------------------------------------

    def add_observation(
        self,
        text: str,
        step: int = 0,
        source: str = "manual",
        tags: Optional[List[str]] = None,
        stream_id: Optional[str] = None,
    ) -> ObservationNode:
        node = ObservationNode(
            text=text, step=step, source=source,
            tags=tags or [], stream_id=stream_id,
        )
        self._nodes[node.node_id] = node
        if stream_id and stream_id in self._streams:
            self._streams[stream_id].observation_ids.append(node.node_id)
        self._append_event("add_observation", node.to_dict(), step=step)
        return node

    def add_hypothesis(
        self,
        condition: str,
        intervention: Dict[str, Any],
        expected_outcome: str,
        step: int = 0,
        stream_id: Optional[str] = None,
        source: str = "manual",
        status: str = "proposed",
        mutation_ref: Optional[str] = None,
    ) -> HypothesisNode:
        # Auto-dilute if unacknowledged shadow mutations exist
        diluted = bool(self._shadow_steps) and status != "shadow"
        node = HypothesisNode(
            condition=condition,
            intervention=intervention,
            expected_outcome=expected_outcome,
            step_created=step,
            stream_id=stream_id,
            source=source,
            status=status,
            mutation_ref=mutation_ref,
            diluted=diluted,
        )
        self._nodes[node.node_id] = node
        if stream_id and stream_id in self._streams:
            self._streams[stream_id].hypothesis_ids.append(node.node_id)
        self._append_event("add_hypothesis", node.to_dict(), step=step)
        return node

    def add_evidence(
        self,
        hypothesis_id: str,
        outcome: str,
        delta: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None,
        effect_id: str = "",
        source: str = "effect_tracker",
        step: int = 0,
    ) -> EvidenceNode:
        # Auto-dilute if unacknowledged shadow mutations exist
        diluted = bool(self._shadow_steps)
        node = EvidenceNode(
            hypothesis_id=hypothesis_id,
            outcome=outcome,
            delta=delta or {},
            context=context or {},
            effect_id=effect_id,
            source=source,
            step=step,
            diluted=diluted,
        )
        self._nodes[node.node_id] = node
        # Update hypothesis evidence_count
        hyp = self._nodes.get(hypothesis_id)
        if isinstance(hyp, HypothesisNode):
            hyp.evidence_count += 1
        # Auto-create edge
        edge_type = "supports" if outcome == "improved" else "contradicts"
        if outcome in ("neutral", "timeout"):
            edge_type = "supports"  # neutral counts as weak support
        self.add_edge(node.node_id, hypothesis_id, edge_type)
        self._append_event("add_evidence", node.to_dict(), step=step)
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
    ) -> ResearchEdge:
        if edge_type not in VALID_EDGE_TYPES:
            raise ValueError(f"Invalid edge type: {edge_type}")
        edge = ResearchEdge(
            source_id=source_id, target_id=target_id, edge_type=edge_type,
        )
        self._edges[edge.edge_id] = edge
        self._append_event("add_edge", edge.to_dict())
        return edge

    def transition_hypothesis(self, hyp_id: str, new_status: str) -> bool:
        node = self._nodes.get(hyp_id)
        if not isinstance(node, HypothesisNode):
            return False
        allowed = HYP_TRANSITIONS.get(node.status, set())
        if new_status not in allowed:
            return False
        old_status = node.status
        node.status = new_status
        self._append_event("transition", {
            "node_id": hyp_id,
            "old_status": old_status,
            "new_status": new_status,
        })
        return True

    def create_stream(self, name: str, description: str = "") -> ResearchStream:
        stream = ResearchStream(name=name, description=description)
        self._streams[stream.stream_id] = stream
        self._append_event("create_stream", stream.to_dict())
        return stream

    def conclude_stream(self, stream_id: str, conclusion: str) -> bool:
        stream = self._streams.get(stream_id)
        if stream is None:
            return False
        if stream.status == "concluded":
            return False
        stream.status = "concluded"
        stream.conclusion = conclusion
        self._append_event("conclude_stream", {
            "stream_id": stream_id,
            "conclusion": conclusion,
        })
        return True

    # ------------------------------------------------------------------
    # Shadow / contamination tracking
    # ------------------------------------------------------------------

    def add_shadow_hypothesis(
        self,
        mutation_ref: str,
        intervention: Dict[str, Any],
        step: int = 0,
        source: str = "shadow",
    ) -> HypothesisNode:
        """Auto-create a shadow hypothesis for an untracked mutation."""
        node = self.add_hypothesis(
            condition=f"untracked mutation: {mutation_ref}",
            intervention=intervention,
            expected_outcome="unknown — awaiting user/AI structured input",
            step=step,
            source=source,
            status="shadow",
            mutation_ref=mutation_ref,
        )
        # Register this step as having an unacknowledged shadow
        if step not in self._shadow_steps:
            self._shadow_steps.append(step)
        self._append_event("shadow_registered", {
            "node_id": node.node_id, "mutation_ref": mutation_ref, "step": step,
        }, step=step)
        return node

    def acknowledge_shadow(self, hyp_id: str) -> bool:
        """Acknowledge a shadow hypothesis, clearing its contamination."""
        node = self._nodes.get(hyp_id)
        if not isinstance(node, HypothesisNode) or node.status != "shadow":
            return False
        # Remove the shadow step from tracking
        step = node.step_created
        if step in self._shadow_steps:
            self._shadow_steps.remove(step)
        self._append_event("shadow_acknowledged", {
            "node_id": hyp_id, "step": step,
        }, step=step)
        return True

    def promote_shadow(self, hyp_id: str, condition: str = "",
                       expected_outcome: str = "") -> bool:
        """Promote a shadow hypothesis to proposed, optionally updating fields."""
        node = self._nodes.get(hyp_id)
        if not isinstance(node, HypothesisNode) or node.status != "shadow":
            return False
        # Acknowledge first (clear contamination)
        self.acknowledge_shadow(hyp_id)
        # Update fields if provided
        if condition:
            node.condition = condition
        if expected_outcome:
            node.expected_outcome = expected_outcome
        node.diluted = False
        return self.transition_hypothesis(hyp_id, "proposed")

    def shadow_hypotheses(self) -> List[HypothesisNode]:
        """Return all shadow (unacknowledged) hypotheses."""
        return [
            n for n in self._nodes.values()
            if isinstance(n, HypothesisNode) and n.status == "shadow"
        ]

    def diluted_nodes(self) -> List:
        """Return all nodes marked as diluted."""
        return [
            n for n in self._nodes.values()
            if getattr(n, "diluted", False)
        ]

    @property
    def has_unacknowledged_shadows(self) -> bool:
        return len(self._shadow_steps) > 0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_node(self, node_id: str):
        return self._nodes.get(node_id)

    def get_stream(self, stream_id: str) -> Optional[ResearchStream]:
        return self._streams.get(stream_id)

    def list_streams(self) -> List[ResearchStream]:
        return list(self._streams.values())

    def hypotheses_by_status(self, status: str) -> List[HypothesisNode]:
        return [
            n for n in self._nodes.values()
            if isinstance(n, HypothesisNode) and n.status == status
        ]

    def all_hypotheses(self) -> List[HypothesisNode]:
        return [n for n in self._nodes.values() if isinstance(n, HypothesisNode)]

    def evidence_for(self, hyp_id: str) -> List[EvidenceNode]:
        return [
            n for n in self._nodes.values()
            if isinstance(n, EvidenceNode) and n.hypothesis_id == hyp_id
        ]

    def edges_for(self, node_id: str) -> List[ResearchEdge]:
        return [
            e for e in self._edges.values()
            if e.source_id == node_id or e.target_id == node_id
        ]

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def stats(self) -> dict:
        obs = sum(1 for n in self._nodes.values() if isinstance(n, ObservationNode))
        hyps = sum(1 for n in self._nodes.values() if isinstance(n, HypothesisNode))
        evs = sum(1 for n in self._nodes.values() if isinstance(n, EvidenceNode))
        status_counts: Dict[str, int] = {}
        for n in self._nodes.values():
            if isinstance(n, HypothesisNode):
                status_counts[n.status] = status_counts.get(n.status, 0) + 1
        diluted_count = sum(1 for n in self._nodes.values() if getattr(n, "diluted", False))
        return {
            "observations": obs,
            "hypotheses": hyps,
            "evidence": evs,
            "edges": len(self._edges),
            "streams": len(self._streams),
            "hypothesis_statuses": status_counts,
            "shadow_count": status_counts.get("shadow", 0),
            "diluted_count": diluted_count,
            "unacknowledged_shadows": len(self._shadow_steps),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _append_event(self, event_type: str, payload: dict, step: int = 0):
        event = {
            "event": event_type,
            "step": step,
            "wall_time": time.time(),
            "payload": payload,
        }
        os.makedirs(self._run_dir, exist_ok=True)
        with open(self._event_log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def save_snapshot(self):
        data = {
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            "edges": {eid: e.to_dict() for eid, e in self._edges.items()},
            "streams": {sid: s.to_dict() for sid, s in self._streams.items()},
            "shadow_steps": self._shadow_steps,
        }
        os.makedirs(self._run_dir, exist_ok=True)
        with open(self._snapshot_path, "w") as f:
            json.dump(data, f, indent=2)

    def load_snapshot(self) -> bool:
        if not os.path.exists(self._snapshot_path):
            return False
        with open(self._snapshot_path) as f:
            data = json.load(f)
        self._nodes.clear()
        self._edges.clear()
        self._streams.clear()
        self._shadow_steps = data.get("shadow_steps", [])
        for nid, nd in data.get("nodes", {}).items():
            cls = _NODE_CLS.get(nd.get("node_type"))
            if cls:
                self._nodes[nid] = cls.from_dict(nd)
        for eid, ed in data.get("edges", {}).items():
            self._edges[eid] = ResearchEdge.from_dict(ed)
        for sid, sd in data.get("streams", {}).items():
            self._streams[sid] = ResearchStream.from_dict(sd)
        return True

    def replay_events(self) -> int:
        """Rebuild state from JSONL event log. Returns event count."""
        if not os.path.exists(self._event_log_path):
            return 0
        self._nodes.clear()
        self._edges.clear()
        self._streams.clear()
        self._shadow_steps.clear()
        count = 0
        with open(self._event_log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                self._apply_event(event)
                count += 1
        return count

    def _apply_event(self, event: dict):
        etype = event["event"]
        p = event["payload"]
        if etype == "add_observation":
            self._nodes[p["node_id"]] = ObservationNode.from_dict(p)
            sid = p.get("stream_id")
            if sid and sid in self._streams:
                self._streams[sid].observation_ids.append(p["node_id"])
        elif etype == "add_hypothesis":
            self._nodes[p["node_id"]] = HypothesisNode.from_dict(p)
            sid = p.get("stream_id")
            if sid and sid in self._streams:
                self._streams[sid].hypothesis_ids.append(p["node_id"])
        elif etype == "add_evidence":
            self._nodes[p["node_id"]] = EvidenceNode.from_dict(p)
            hyp = self._nodes.get(p.get("hypothesis_id"))
            if isinstance(hyp, HypothesisNode):
                hyp.evidence_count += 1
        elif etype == "add_edge":
            self._edges[p["edge_id"]] = ResearchEdge.from_dict(p)
        elif etype == "transition":
            node = self._nodes.get(p["node_id"])
            if isinstance(node, HypothesisNode):
                node.status = p["new_status"]
        elif etype == "create_stream":
            self._streams[p["stream_id"]] = ResearchStream.from_dict(p)
        elif etype == "conclude_stream":
            stream = self._streams.get(p["stream_id"])
            if stream:
                stream.status = "concluded"
                stream.conclusion = p.get("conclusion", "")
        elif etype == "shadow_registered":
            step = p.get("step", 0)
            if step not in self._shadow_steps:
                self._shadow_steps.append(step)
        elif etype == "shadow_acknowledged":
            step = p.get("step", 0)
            if step in self._shadow_steps:
                self._shadow_steps.remove(step)

    # ------------------------------------------------------------------
    # Cytoscape.js serialization
    # ------------------------------------------------------------------

    def to_cytoscape(self, stream_id: Optional[str] = None) -> dict:
        """Return Cytoscape.js elements format."""
        elements: List[dict] = []

        for nid, node in self._nodes.items():
            # Filter by stream if requested
            if stream_id:
                node_stream = getattr(node, "stream_id", None)
                if isinstance(node, EvidenceNode):
                    hyp = self._nodes.get(node.hypothesis_id)
                    node_stream = getattr(hyp, "stream_id", None) if hyp else None
                if node_stream != stream_id:
                    continue
            nd = node.to_dict()
            nd["id"] = nid
            elements.append({"group": "nodes", "data": nd})

        # Collect visible node IDs for edge filtering
        visible = {e["data"]["id"] for e in elements}

        for eid, edge in self._edges.items():
            if edge.source_id in visible and edge.target_id in visible:
                elements.append({
                    "group": "edges",
                    "data": {
                        "id": eid,
                        "source": edge.source_id,
                        "target": edge.target_id,
                        "edge_type": edge.edge_type,
                    },
                })

        return {"elements": elements}
