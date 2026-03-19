"""Export, import, and merge research graphs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .graph import ResearchGraph


def export_graph(
    graph: ResearchGraph,
    out_path: str,
    stream_id: Optional[str] = None,
) -> str:
    """Export graph (or a single stream) to a JSON file."""
    nodes = {}
    edges = {}

    if stream_id:
        stream = graph.get_stream(stream_id)
        if not stream:
            raise ValueError(f"Stream not found: {stream_id}")
        # Collect nodes in stream
        for nid in stream.hypothesis_ids + stream.observation_ids:
            node = graph.get_node(nid)
            if node:
                nodes[nid] = node.to_dict()
        # Also collect evidence for those hypotheses
        for nid in list(nodes):
            for ev in graph.evidence_for(nid):
                nodes[ev.node_id] = ev.to_dict()
        # Collect edges between visible nodes
        for eid, edge in graph._edges.items():
            if edge.source_id in nodes and edge.target_id in nodes:
                edges[eid] = edge.to_dict()
        streams = {stream_id: stream.to_dict()}
    else:
        nodes = {nid: n.to_dict() for nid, n in graph._nodes.items()}
        edges = {eid: e.to_dict() for eid, e in graph._edges.items()}
        streams = {sid: s.to_dict() for sid, s in graph._streams.items()}

    payload = {
        "version": 1,
        "exported_at": time.time(),
        "source_run_dir": graph._run_dir,
        "streams": list(streams.values()),
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def import_graph(graph: ResearchGraph, path: str) -> int:
    """Replace graph contents from an export file. Returns node count."""
    with open(path) as f:
        data = json.load(f)
    graph._nodes.clear()
    graph._edges.clear()
    graph._streams.clear()
    return _load_export_data(graph, data)


def merge_graph(graph: ResearchGraph, path: str) -> int:
    """Merge nodes/edges from an export file. Skips duplicates. Returns new count."""
    with open(path) as f:
        data = json.load(f)
    return _load_export_data(graph, data, skip_existing=True)


def merge_from_run(graph: ResearchGraph, other_run_dir: str) -> int:
    """Merge from another run's snapshot."""
    other = ResearchGraph(other_run_dir)
    loaded = other.load_snapshot()
    if not loaded:
        count = other.replay_events()
        if count == 0:
            raise ValueError(f"No research data in {other_run_dir}")

    # Export other to a temp dict and merge
    data = {
        "nodes": [n.to_dict() for n in other._nodes.values()],
        "edges": [e.to_dict() for e in other._edges.values()],
        "streams": [s.to_dict() for s in other._streams.values()],
    }
    return _load_export_data(graph, data, skip_existing=True)


def _load_export_data(
    graph: ResearchGraph,
    data: dict,
    skip_existing: bool = False,
) -> int:
    from .types import ObservationNode, HypothesisNode, EvidenceNode, ResearchEdge, ResearchStream

    _NODE_CLS = {
        "observation": ObservationNode,
        "hypothesis": HypothesisNode,
        "evidence": EvidenceNode,
    }
    count = 0

    for sd in data.get("streams", []):
        sid = sd.get("stream_id")
        if skip_existing and sid in graph._streams:
            continue
        graph._streams[sid] = ResearchStream.from_dict(sd)
        count += 1

    for nd in data.get("nodes", []):
        nid = nd.get("node_id")
        if skip_existing and nid in graph._nodes:
            continue
        cls = _NODE_CLS.get(nd.get("node_type"))
        if cls:
            graph._nodes[nid] = cls.from_dict(nd)
            count += 1

    for ed in data.get("edges", []):
        eid = ed.get("edge_id")
        if skip_existing and eid in graph._edges:
            continue
        graph._edges[eid] = ResearchEdge.from_dict(ed)
        count += 1

    return count
