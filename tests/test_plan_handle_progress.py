"""Tests for PlanHandle.get_progress() and PlanProgress."""

import json
import tempfile
from pathlib import Path

import pytest

from muflow import TaskNode, TaskPlan
from muflow.backends import LocalBackend
from muflow.backends.handle import PlanHandle, PlanProgress
from muflow.registry import TaskEntry


# ── PlanProgress model ────────────────────────────────────────────────────────


class TestPlanProgress:
    def test_fraction_all_complete(self):
        p = PlanProgress(total=4, completed=4, node_breakdown={})
        assert p.fraction == 1.0

    def test_fraction_partial(self):
        p = PlanProgress(total=4, completed=2, node_breakdown={})
        assert p.fraction == 0.5

    def test_fraction_none_complete(self):
        p = PlanProgress(total=4, completed=0, node_breakdown={})
        assert p.fraction == 0.0

    def test_fraction_empty_plan(self):
        p = PlanProgress(total=0, completed=0, node_breakdown={})
        assert p.fraction == 0.0

    def test_is_complete_true(self):
        p = PlanProgress(total=3, completed=3, node_breakdown={})
        assert p.is_complete is True

    def test_is_complete_false(self):
        p = PlanProgress(total=3, completed=2, node_breakdown={})
        assert p.is_complete is False


# ── get_progress with LocalProgressChecker ───────────────────────────────────


class TestGetProgressLocal:
    def _make_handle(self, node_prefixes: dict) -> PlanHandle:
        return PlanHandle(
            backend="local",
            plan_id="test-plan",
            node_prefixes=node_prefixes,
            storage_type="local",
            storage_config={},
        )

    def test_all_complete(self, tmp_path):
        prefixes = {}
        for name in ("a", "b", "c"):
            d = tmp_path / name
            d.mkdir()
            (d / "manifest.json").write_text("{}")
            prefixes[name] = str(d)

        progress = self._make_handle(prefixes).get_progress()
        assert progress.total == 3
        assert progress.completed == 3
        assert progress.is_complete is True
        assert all(progress.node_breakdown.values())

    def test_partial_completion(self, tmp_path):
        done = tmp_path / "done"
        pending = tmp_path / "pending"
        done.mkdir()
        pending.mkdir()
        (done / "manifest.json").write_text("{}")

        prefixes = {"done": str(done), "pending": str(pending)}
        progress = self._make_handle(prefixes).get_progress()
        assert progress.total == 2
        assert progress.completed == 1
        assert progress.fraction == 0.5
        assert progress.node_breakdown["done"] is True
        assert progress.node_breakdown["pending"] is False

    def test_none_complete(self, tmp_path):
        d = tmp_path / "node"
        d.mkdir()
        progress = self._make_handle({"n": str(d)}).get_progress()
        assert progress.completed == 0
        assert progress.is_complete is False

    def test_empty_plan(self):
        progress = self._make_handle({}).get_progress()
        assert progress.total == 0
        assert progress.completed == 0
        assert progress.fraction == 0.0


# ── Serialization round-trip ──────────────────────────────────────────────────


class TestPlanHandleSerialization:
    def test_roundtrip_preserves_node_prefixes(self, tmp_path):
        handle = PlanHandle(
            backend="local",
            plan_id="root-key",
            node_prefixes={"n1": "/tmp/n1", "n2": "/tmp/n2"},
            storage_type="local",
            storage_config={},
        )
        restored = PlanHandle.from_json(handle.to_json())
        assert restored.node_prefixes == handle.node_prefixes
        assert restored.storage_type == "local"
        assert restored.storage_config == {}

    def test_roundtrip_s3_config(self):
        handle = PlanHandle(
            backend="celery",
            plan_id="celery-uuid",
            node_prefixes={"n": "muflow/task/aaa"},
            storage_type="s3",
            storage_config={"bucket": "my-bucket"},
        )
        restored = PlanHandle.from_json(handle.to_json())
        assert restored.storage_config == {"bucket": "my-bucket"}
        assert restored.storage_type == "s3"

    def test_get_progress_after_deserialization(self, tmp_path):
        d = tmp_path / "node"
        d.mkdir()
        (d / "manifest.json").write_text("{}")

        handle = PlanHandle(
            backend="local",
            plan_id="p",
            node_prefixes={"n": str(d)},
            storage_type="local",
            storage_config={},
        )
        restored = PlanHandle.from_json(handle.to_json())
        progress = restored.get_progress()
        assert progress.is_complete is True


# ── Integration: LocalBackend.submit_plan() → PlanHandle.get_progress() ───────


def _noop(ctx):
    ctx.save_json("result.json", {"done": True})


_REGISTRY = {"test.progress_task": TaskEntry(name="test.progress_task", fn=_noop)}


class TestLocalBackendProgress:
    def test_handle_has_node_prefixes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            node = TaskNode(
                key="n1",
                function="test.progress_task",
                subject_key="test:1",
                kwargs={},
                storage_prefix=f"{tmpdir}/n1",
                depends_on=[],
            )
            plan = TaskPlan(nodes={"n1": node}, root_key="n1")
            backend = LocalBackend(tmpdir, _REGISTRY.get)
            handle = backend.submit_plan(plan)

            assert handle.storage_type == "local"
            assert handle.storage_config == {}
            assert set(handle.node_prefixes.keys()) == {"n1"}
            assert handle.node_prefixes["n1"] == f"{tmpdir}/n1"

    def test_get_progress_complete_after_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            node = TaskNode(
                key="n1",
                function="test.progress_task",
                subject_key="test:1",
                kwargs={},
                storage_prefix=f"{tmpdir}/n1",
                depends_on=[],
            )
            plan = TaskPlan(nodes={"n1": node}, root_key="n1")
            handle = LocalBackend(tmpdir, _REGISTRY.get).submit_plan(
                TaskPlan(nodes={"n1": node}, root_key="n1")
            )

            progress = handle.get_progress()
            assert progress.is_complete is True
            assert progress.completed == 1
            assert progress.total == 1

    def test_get_progress_multi_node(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nodes = {}
            for i in range(3):
                node = TaskNode(
                    key=f"n{i}",
                    function="test.progress_task",
                    subject_key=f"test:{i}",
                    kwargs={},
                    storage_prefix=f"{tmpdir}/n{i}",
                    depends_on=[],
                )
                nodes[f"n{i}"] = node

            plan = TaskPlan(nodes=nodes, root_key="n0")
            handle = LocalBackend(tmpdir, _REGISTRY.get).submit_plan(plan)

            progress = handle.get_progress()
            assert progress.total == 3
            assert progress.completed == 3
            assert progress.is_complete is True
            assert all(progress.node_breakdown.values())
