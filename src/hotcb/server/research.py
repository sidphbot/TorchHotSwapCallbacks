"""REST router for the research & learning module (closure-based factory)."""

from typing import Any, Optional


def create_router(engine: Any) -> Any:
    """Factory: build research API router closed over a ResearchEngine instance."""
    from fastapi import APIRouter
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    import os as _os
    _engine = engine
    _last_snapshot_mtime = [0.0]  # mutable container for closure
    router = APIRouter(prefix="/api/research", tags=["research"])

    def _maybe_reload():
        """Reload graph from disk if snapshot file was modified externally."""
        try:
            snap = _engine.graph._snapshot_path
            if _os.path.exists(snap):
                mt = _os.path.getmtime(snap)
                if mt > _last_snapshot_mtime[0]:
                    _engine.graph.load_snapshot()
                    _last_snapshot_mtime[0] = mt
        except Exception:
            pass

    # ----- Pydantic request models -----

    class ObserveRequest(BaseModel):
        text: str
        step: int = 0
        tags: list = []
        stream_id: Optional[str] = None

    class HypothesizeRequest(BaseModel):
        condition: str
        intervention: dict
        expected_outcome: str
        stream_id: Optional[str] = None

    class RefineRequest(BaseModel):
        condition: str
        intervention: dict
        expected_outcome: str

    class ConcludeHypRequest(BaseModel):
        status: str
        reason: str = ""

    class StreamRequest(BaseModel):
        name: str
        description: str = ""

    class ConcludeStreamRequest(BaseModel):
        conclusion: str

    class PredictRequest(BaseModel):
        intervention: dict
        context: dict = {}

    class SuggestRequest(BaseModel):
        context: dict = {}
        top_k: int = 3

    class ImportRequest(BaseModel):
        path: str

    class MergeRequest(BaseModel):
        run_dir: str

    class ExportRequest(BaseModel):
        out: str
        stream_id: Optional[str] = None

    class ExportRecipeRequest(BaseModel):
        out: Optional[str] = None

    class PromoteShadowRequest(BaseModel):
        condition: str = ""
        expected_outcome: str = ""

    class LoadConditionsRequest(BaseModel):
        task: str
        base_run_dir: str = ""

    class LaunchTestsRequest(BaseModel):
        hypothesis_ids: list
        base_run_dir: str

    # ----- Graph endpoints -----

    @router.get("/graph")
    async def get_graph(stream_id: Optional[str] = None):
        _maybe_reload()
        return _engine.graph.to_cytoscape(stream_id=stream_id)

    @router.get("/stats")
    async def get_stats():
        _maybe_reload()
        return _engine.graph.stats()

    # ----- Streams -----

    @router.get("/streams")
    async def list_streams():
        return {"streams": [s.to_dict() for s in _engine.graph.list_streams()]}

    @router.post("/streams")
    async def create_stream(body: StreamRequest):
        s = _engine.graph.create_stream(body.name, body.description)
        return s.to_dict()

    @router.post("/streams/{stream_id}/conclude")
    async def conclude_stream(stream_id: str, body: ConcludeStreamRequest):
        ok = _engine.graph.conclude_stream(stream_id, body.conclusion)
        if not ok:
            return JSONResponse(status_code=400, content={"detail": "Cannot conclude stream"})
        return {"status": "concluded", "stream_id": stream_id}

    # ----- Observations -----

    @router.post("/observe")
    async def observe(body: ObserveRequest):
        node = _engine.observe(
            text=body.text, step=body.step,
            tags=body.tags, stream_id=body.stream_id,
        )
        return node.to_dict()

    # ----- Hypotheses -----

    @router.post("/hypothesize")
    async def hypothesize(body: HypothesizeRequest):
        node = _engine.hypothesize(
            condition=body.condition,
            intervention=body.intervention,
            expected_outcome=body.expected_outcome,
            stream_id=body.stream_id,
        )
        return node.to_dict()

    @router.post("/refine/{obs_id}")
    async def refine(obs_id: str, body: RefineRequest):
        try:
            node = _engine.refine(
                obs_id,
                condition=body.condition,
                intervention=body.intervention,
                expected_outcome=body.expected_outcome,
            )
            return node.to_dict()
        except ValueError as e:
            return JSONResponse(status_code=404, content={"detail": str(e)})

    @router.get("/hypotheses")
    async def list_hypotheses(status: Optional[str] = None):
        if status:
            hyps = _engine.graph.hypotheses_by_status(status)
        else:
            hyps = _engine.graph.all_hypotheses()
        return {"hypotheses": [h.to_dict() for h in hyps]}

    @router.get("/hypotheses/{hyp_id}")
    async def get_hypothesis(hyp_id: str):
        node = _engine.graph.get_node(hyp_id)
        if node is None:
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        return node.to_dict()

    @router.post("/hypotheses/{hyp_id}/test")
    async def test_hypothesis(hyp_id: str):
        try:
            result = _engine.test_hypothesis(hyp_id)
            return result
        except ValueError as e:
            return JSONResponse(status_code=400, content={"detail": str(e)})

    @router.post("/hypotheses/{hyp_id}/conclude")
    async def conclude_hypothesis(hyp_id: str, body: ConcludeHypRequest):
        ok = _engine.conclude_hypothesis(hyp_id, body.status, body.reason)
        if not ok:
            return JSONResponse(status_code=400, content={"detail": "Invalid transition"})
        return {"status": body.status, "hypothesis_id": hyp_id}

    # ----- Evidence -----

    @router.get("/evidence/{hyp_id}")
    async def get_evidence(hyp_id: str):
        evs = _engine.graph.evidence_for(hyp_id)
        return {"evidence": [e.to_dict() for e in evs]}

    # ----- Shadow / contamination -----

    @router.get("/shadows")
    async def list_shadows():
        shadows = _engine.graph.shadow_hypotheses()
        return {"shadows": [s.to_dict() for s in shadows]}

    @router.get("/contamination")
    async def get_contamination():
        return _engine.contamination_status

    @router.post("/shadows/{hyp_id}/acknowledge")
    async def acknowledge_shadow(hyp_id: str):
        ok = _engine.acknowledge_shadow(hyp_id)
        if not ok:
            return JSONResponse(status_code=400, content={"detail": "Not a shadow hypothesis"})
        return {"status": "acknowledged", "hypothesis_id": hyp_id}

    @router.post("/shadows/{hyp_id}/promote")
    async def promote_shadow(hyp_id: str, body: PromoteShadowRequest):
        ok = _engine.promote_shadow(hyp_id, body.condition, body.expected_outcome)
        if not ok:
            return JSONResponse(status_code=400, content={"detail": "Cannot promote"})
        return {"status": "promoted", "hypothesis_id": hyp_id}

    @router.get("/diluted")
    async def list_diluted():
        diluted = _engine.graph.diluted_nodes()
        return {"diluted": [d.to_dict() for d in diluted]}

    # ----- Continuation bridge -----

    @router.post("/load-conditions")
    async def load_conditions(body: LoadConditionsRequest):
        hyps = _engine.load_continuation_conditions(body.task, body.base_run_dir)
        return {"loaded": len(hyps), "hypotheses": [h.to_dict() for h in hyps]}

    @router.post("/launch-tests")
    async def launch_tests(body: LaunchTestsRequest):
        import asyncio
        results = await asyncio.to_thread(
            _engine.launch_condition_tests, body.hypothesis_ids, body.base_run_dir
        )
        return {"results": results}

    # ----- NN endpoints -----

    @router.get("/nn/status")
    async def nn_status():
        learner = _engine._get_learner()
        if learner is None:
            return {"loaded": False, "nn_mode": _engine._nn_mode}
        return learner.status

    @router.post("/nn/predict")
    async def nn_predict(body: PredictRequest):
        score = _engine.predict_outcome(body.intervention, body.context)
        return {"predicted_p_improved": score}

    @router.post("/nn/suggest")
    async def nn_suggest(body: SuggestRequest):
        suggestions = _engine.suggest_interventions(body.context, top_k=body.top_k)
        return {"suggestions": suggestions}

    @router.post("/nn/discover")
    async def nn_discover():
        patterns = _engine.discover_patterns()
        return {"discovered": [h.to_dict() for h in patterns]}

    # ----- Export / Import / Merge -----

    @router.post("/export")
    async def export_graph(body: ExportRequest):
        from hotcb.research.export import export_graph as _export
        path = _export(_engine.graph, body.out, stream_id=body.stream_id)
        return {"path": path}

    @router.post("/import")
    async def import_graph(body: ImportRequest):
        from hotcb.research.export import import_graph as _import
        count = _import(_engine.graph, body.path)
        return {"imported": count}

    @router.post("/merge")
    async def merge_graph(body: MergeRequest):
        from hotcb.research.export import merge_from_run
        count = merge_from_run(_engine.graph, body.run_dir)
        return {"merged": count}

    @router.post("/export-recipe")
    async def export_recipe(body: ExportRecipeRequest):
        path = _engine.export_confirmed_recipe(out_path=body.out)
        return {"path": path}

    return router
