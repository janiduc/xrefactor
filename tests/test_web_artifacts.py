"""
Tests for the saved-artifact and benchmark API routes.

These exist because the UI previously had no way to see anything without
re-running a multi-minute pipeline: the old `/api/results/<id>` route read
`outputs/<id>/results.json`, a layout the pipeline has never written, so it
404'd for every input while ~30 saved runs sat unreachable on disk. The
evidence-card report was in the same position - written by stage 4, served by
nothing.

The routes are backed by real files in `outputs/`, so each test skips rather
than fails when the fixture data is absent (a fresh clone has no runs yet).
That keeps the suite meaningful on a populated checkout without making it
depend on artifacts that are not in version control.
"""

import json
import os
from pathlib import Path

import pytest

from src.web import create_app


OUTPUTS = Path("./outputs")


@pytest.fixture
def client():
    return create_app({"TESTING": True}).test_client()


def _saved_run_ids():
    return sorted((p.stem[len("pipeline_results_"):]
                   for p in OUTPUTS.glob("pipeline_results_*.json")), reverse=True)


@pytest.fixture
def run_id():
    ids = _saved_run_ids()
    if not ids:
        pytest.skip("no saved pipeline runs in outputs/")
    return ids[0]


class TestRunListing:
    def test_lists_saved_runs_newest_first(self, client):
        response = client.get("/api/runs")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert data["count"] == len(data["runs"])
        ids = [r["id"] for r in data["runs"]]
        assert ids == sorted(ids, reverse=True)

    def test_summary_projects_fields_without_the_payload(self, client, run_id):
        """A listing must not inline every run's code strings - the newest
        result is ~1MB, almost all of it original/refactored source."""
        row = next(r for r in client.get("/api/runs").get_json()["runs"] if r["id"] == run_id)
        for key in ("id", "run_status", "total_nodes", "suggestion_count",
                    "ast_count", "neural_count", "has_evidence"):
            assert key in row
        assert "stages" not in row
        assert row["ast_count"] + row["neural_count"] == row["suggestion_count"]

    def test_unlabelled_runs_report_null_rather_than_guessing(self, client):
        """Runs saved before the pipeline recorded `data_directory` have no
        corpus label. That must surface as null, not an invented value."""
        runs = client.get("/api/runs").get_json()["runs"]
        for row in runs:
            assert "repo_label" in row
            assert row["repo_label"] is None or isinstance(row["repo_label"], str)


class TestRunFetch:
    def test_fetches_one_run(self, client, run_id):
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        data = response.get_json()
        assert data["id"] == run_id
        assert "stages" in data["results"]

    def test_latest_alias_resolves(self, client):
        response = client.get("/api/runs/latest")
        if not _saved_run_ids():
            assert response.status_code == 404
            return
        assert response.status_code == 200
        assert response.get_json()["id"] == _saved_run_ids()[0]

    def test_unknown_run_is_404(self, client):
        response = client.get("/api/runs/not-a-timestamp")
        assert response.status_code == 404
        assert "not found" in response.get_json()["message"].lower()

    def test_traversal_attempt_does_not_escape_outputs(self, client):
        """Ids are matched against the known set rather than joined into a
        path, so a traversal attempt cannot reach outside outputs/."""
        for attempt in ("../config", "..%2F..%2Fconfigs%2Fconfig", "....//config"):
            assert client.get(f"/api/runs/{attempt}").status_code in (404, 308)

    def test_deprecated_results_alias_still_resolves(self, client, run_id):
        """The old route is kept as an alias so existing links keep working -
        but it now actually returns the run instead of always 404ing."""
        assert client.get(f"/api/results/{run_id}").status_code == 200


class TestEvidence:
    def test_serves_the_evidence_report_when_present(self, client):
        candidates = [p for p in OUTPUTS.glob("evidence_cards_*.json")]
        if not candidates:
            pytest.skip("no evidence card files in outputs/")
        run = sorted(candidates, reverse=True)[0].stem[len("evidence_cards_"):]
        response = client.get(f"/api/runs/{run}/evidence")
        assert response.status_code == 200
        report = response.get_json()["report"]
        assert "evidence_cards" in report
        assert "risk_assessment" in report

    def test_run_without_evidence_is_404_not_500(self, client):
        """Older runs predate the evidence-cards file; that is a legitimate
        absence, not a server error."""
        without = [rid for rid in _saved_run_ids()
                   if not (OUTPUTS / f"evidence_cards_{rid}.json").exists()]
        if not without:
            pytest.skip("every saved run has evidence cards")
        assert client.get(f"/api/runs/{without[0]}/evidence").status_code == 404


class TestBenchmarks:
    def test_taxonomy_comes_from_the_pipeline_constants(self, client):
        """The scorecard's engine column depends on this: duplicating the
        pattern list in JS would let it rot when the engine gains a pattern."""
        from src.gnn.refactoring_types import REFACTORING_TYPE_TO_ID
        from src.refactor.engine import SUPPORTED_PATTERNS

        taxonomy = client.get("/api/benchmarks").get_json()["taxonomy"]
        expected_order = [name for name, _ in
                          sorted(REFACTORING_TYPE_TO_ID.items(), key=lambda kv: kv[1])]
        assert taxonomy["order"] == expected_order
        assert taxonomy["ast_supported"] == sorted(SUPPORTED_PATTERNS)
        assert taxonomy["ids"] == dict(REFACTORING_TYPE_TO_ID)

    def test_every_payload_carries_provenance(self, client):
        """Offline evaluation describes a different corpus and date than any
        run. Without provenance the UI cannot stop the two being read as one
        experiment."""
        data = client.get("/api/benchmarks").get_json()
        for section in ("eval", "mining"):
            if data.get(section):
                prov = data[section]["provenance"]
                assert prov["source_file"]
                assert prov["mtime_iso"]
        if data.get("eval"):
            # The repo the end-to-end trace ran on drives the mismatch warning.
            assert "repo" in data["eval"]["provenance"]
        for model in data.get("models", {}).values():
            assert model["provenance"]["source_file"]

    def test_does_not_serve_the_duplicate_baseline_file(self, client):
        """outputs/baseline_prefix_*.json is byte-identical to the evaluation
        report and referenced by no code. Serving it would invite a fabricated
        'vs. baseline' comparison, so it must stay out of the payload."""
        body = json.dumps(client.get("/api/benchmarks").get_json())
        assert "baseline_prefix" not in body

    def test_classifier_and_generation_sections_are_passed_through(self, client):
        data = client.get("/api/benchmarks").get_json()
        if not data.get("eval"):
            pytest.skip("no evaluation report in outputs/")
        assert "per_class" in data["eval"]["classifier"]
        assert "per_pattern" in data["eval"]["generation"]

    def test_responds_even_with_no_artifacts(self, client, monkeypatch, tmp_path):
        """A fresh checkout has no outputs/ and no models/. The endpoint must
        still answer with the taxonomy rather than erroring."""
        from src.web import api as api_module
        monkeypatch.setattr(api_module, "OUTPUTS_DIR", str(tmp_path / "nope"))
        monkeypatch.setattr(api_module, "_MODELS_DIR", str(tmp_path / "nope"))
        monkeypatch.setattr(api_module, "_MINING_REPORT", str(tmp_path / "nope.json"))
        response = client.get("/api/benchmarks")
        assert response.status_code == 200
        data = response.get_json()
        assert data["taxonomy"]["order"]
        assert data["eval"] is None
        assert data["models"] == {}


class TestPipelineRecordsProvenance:
    def test_run_pipeline_records_the_analysed_directory(self):
        """`data_directory`/`repo_label` are what let a saved run be labelled
        and compared against the evaluation corpus. Asserted on the source of
        truth rather than by running the pipeline, which takes minutes."""
        source = Path("src/core/pipeline.py").read_text(encoding="utf-8")
        assert '"data_directory": data_directory' in source
        assert '"repo_label"' in source
