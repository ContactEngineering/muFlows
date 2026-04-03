"""Tests for muflow.testing utilities."""

import tempfile
from pathlib import Path

import pytest

from muflow import register_workflow, run_plan_locally
from muflow.pipeline import ForEach, Pipeline, Step
from muflow.registry import get, unregister


@pytest.fixture(autouse=True)
def register_test_workflows():
    """Register test workflows for each test."""
    for name in [
        "test.testing.simple_workflow",
        "test.testing.workflow_with_deps",
        "test.testing.failing_workflow",
    ]:
        if get(name) is not None:
            unregister(name)

    @register_workflow(name="test.testing.simple_workflow")
    def simple_workflow(context):
        """Simple workflow that writes a result."""
        params = context.kwargs
        context.save_json("result.json", {
            "id": params.get("id", "unknown"),
            "status": "completed",
        })

    @register_workflow(name="test.testing.workflow_with_deps")
    def workflow_with_deps(context):
        """Workflow that reads from dependencies."""
        dep_results = []
        for key in context.dependency_keys():
            dep = context.dependency(key)
            result = dep.read_json("result.json")
            dep_results.append(result["id"])

        context.save_json("combined.json", {
            "dependencies": dep_results,
            "status": "completed",
        })

    @register_workflow(name="test.testing.failing_workflow")
    def failing_test_workflow(context):
        """Workflow that always fails."""
        raise ValueError("Intentional test failure")

    yield


# ── Pipelines for testing ─────────────────────────────────────────────────

def _simple_pipeline():
    return Pipeline(
        name="test.simple_pipeline",
        steps={
            "simple": Step(workflow="test.testing.simple_workflow"),
        },
    )


def _pipeline_with_deps():
    return Pipeline(
        name="test.deps_pipeline",
        steps={
            "deps": ForEach(
                workflow="test.testing.simple_workflow",
                over=lambda sk, kw: [
                    {"id": f"dep_{i}"}
                    for i in range(kw.get("num_deps", 2))
                ],
            ),
            "main": Step(
                workflow="test.testing.workflow_with_deps",
                after=["deps"],
            ),
        },
    )


def _failing_pipeline():
    return Pipeline(
        name="test.failing_pipeline",
        steps={
            "fail": Step(workflow="test.testing.failing_workflow"),
        },
    )


class TestRunPlanLocally:
    """Tests for run_plan_locally function."""

    def test_run_simple_workflow(self):
        """Should execute a simple workflow and return result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_plan_locally(
                pipeline=_simple_pipeline(),
                subject_key="test:1",
                kwargs={"id": "test_node"},
                output_dir=tmpdir,
            )

            assert result.success
            assert result.error is None
            assert result.output_dir == Path(tmpdir)
            assert result.root_output_dir.exists()

    def test_read_json_output(self):
        """Should be able to read JSON output from result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_plan_locally(
                pipeline=_simple_pipeline(),
                subject_key="test:1",
                kwargs={"id": "my_test_id"},
                output_dir=tmpdir,
            )

            assert result.success
            data = result.read_json("result.json")
            assert data["id"] == "my_test_id"
            assert data["status"] == "completed"

    def test_list_files(self):
        """Should list files in output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_plan_locally(
                pipeline=_simple_pipeline(),
                subject_key="test:1",
                kwargs={"id": "test"},
                output_dir=tmpdir,
            )

            files = result.list_files()
            assert "result.json" in files

    def test_workflow_with_dependencies(self):
        """Should execute workflow with dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_plan_locally(
                pipeline=_pipeline_with_deps(),
                subject_key="test:root",
                kwargs={"num_deps": 3},
                output_dir=tmpdir,
            )

            assert result.success
            assert len(result.plan.nodes) == 4  # 3 deps + 1 main

            data = result.read_json("combined.json")
            assert data["status"] == "completed"
            assert len(data["dependencies"]) == 3

    def test_failed_workflow(self):
        """Should handle workflow failure gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_plan_locally(
                pipeline=_failing_pipeline(),
                subject_key="test:1",
                kwargs={},
                output_dir=tmpdir,
            )

            assert not result.success
            assert result.error is not None
            assert "Intentional test failure" in result.error

    def test_verbose_mode(self, capsys):
        """Should print progress in verbose mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_plan_locally(
                pipeline=_simple_pipeline(),
                subject_key="test:1",
                kwargs={"id": "test"},
                output_dir=tmpdir,
                verbose=True,
            )

            assert result.success
            captured = capsys.readouterr()
            assert "Planning" in captured.out
            assert "Executing" in captured.out

    def test_read_file_bytes(self):
        """Should be able to read file as bytes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_plan_locally(
                pipeline=_simple_pipeline(),
                subject_key="test:1",
                kwargs={"id": "test"},
                output_dir=tmpdir,
            )

            data = result.read_file("result.json")
            assert isinstance(data, bytes)
            assert b"test" in data
