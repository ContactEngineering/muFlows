"""Tests for execution backends."""

import tempfile
from pathlib import Path

import pytest

from muflow import TaskNode, TaskPlan, register_task, registry
from muflow.backends import LocalBackend


# Test tasks
@register_task(name="test.leaf_task")
def leaf_task(context):
    """Simple leaf task that writes a result."""
    params = context.kwargs
    context.save_json("result.json", {
        "id": params.get("id", "unknown"),
        "status": "completed",
    })


@register_task(name="test.dependent_task")
def dependent_task(context):
    """Task that reads from dependencies."""
    # Read from dependencies
    dep_results = []
    for i in range(3):
        dep_key = f"dep_{i}"
        if context.has_dependency(dep_key):
            dep = context.dependency(dep_key)
            result = dep.read_json("result.json")
            dep_results.append(result["id"])

    context.save_json("result.json", {
        "dependencies": dep_results,
        "status": "completed",
    })


@register_task(name="test.failing_task")
def failing_task(context):
    """Task that always fails."""
    raise ValueError("Intentional failure for testing")


class TestLocalBackend:
    """Tests for LocalBackend."""

    def test_execute_single_node_plan(self):
        """Should execute a plan with a single node."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build a simple plan with one node
            node = TaskNode(
                key="node1",
                function="test.leaf_task",
                subject_key="test:1",
                kwargs={"id": "node1"},
                storage_prefix=f"{tmpdir}/node1",
                depends_on=[],
            )
            plan = TaskPlan(nodes={"node1": node}, root_key="node1")

            # Execute
            backend = LocalBackend(tmpdir, registry.get)
            handle = backend.submit_plan(plan)

            # Verify
            assert backend.get_plan_state(handle.plan_id) == "success"
            result_path = Path(tmpdir) / "node1" / "result.json"
            assert result_path.exists()

    def test_execute_plan_with_dependencies(self):
        """Should execute nodes in dependency order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build a plan: 3 leaf nodes -> 1 dependent node
            nodes = {}

            # Leaf nodes
            for i in range(3):
                node = TaskNode(
                    key=f"leaf_{i}",
                    function="test.leaf_task",
                    subject_key=f"test:{i}",
                    kwargs={"id": f"leaf_{i}"},
                    storage_prefix=f"{tmpdir}/leaf_{i}",
                    depends_on=[],
                )
                nodes[node.key] = node

            # Dependent node
            root = TaskNode(
                key="root",
                function="test.dependent_task",
                subject_key="test:root",
                kwargs={},
                storage_prefix=f"{tmpdir}/root",
                depends_on=["leaf_0", "leaf_1", "leaf_2"],
            )
            nodes["root"] = root

            plan = TaskPlan(nodes=nodes, root_key="root")

            # Execute
            backend = LocalBackend(tmpdir, registry.get)

            completed_nodes = []

            def on_complete(key):
                completed_nodes.append(key)

            handle = backend.submit_plan(plan, on_node_complete=on_complete)

            # Verify execution order
            assert backend.get_plan_state(handle.plan_id) == "success"
            assert len(completed_nodes) == 4

            # Leaf nodes should complete before root
            root_idx = completed_nodes.index("root")
            for i in range(3):
                leaf_idx = completed_nodes.index(f"leaf_{i}")
                assert leaf_idx < root_idx, "Leaf should complete before root"

    def test_execute_plan_with_cached_nodes(self):
        """Nodes with existing manifest.json are detected as cached at execution time."""
        import json as _json

        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create results for one node, including manifest.json
            cached_dir = Path(tmpdir) / "cached"
            cached_dir.mkdir()
            (cached_dir / "result.json").write_text(
                '{"id": "cached", "status": "completed"}'
            )
            (cached_dir / "manifest.json").write_text(
                _json.dumps({"files": ["result.json"], "timestamp": "2024-01-01T00:00:00+00:00"})
            )

            # Build plan — no cached=True field any more; caching is detected at runtime
            cached_node = TaskNode(
                key="cached",
                function="test.leaf_task",
                subject_key="test:cached",
                kwargs={"id": "cached"},
                storage_prefix=str(cached_dir),
                depends_on=[],
            )
            new_node = TaskNode(
                key="new",
                function="test.leaf_task",
                subject_key="test:new",
                kwargs={"id": "new"},
                storage_prefix=f"{tmpdir}/new",
                depends_on=[],
            )
            root = TaskNode(
                key="root",
                function="test.dependent_task",
                subject_key="test:root",
                kwargs={},
                storage_prefix=f"{tmpdir}/root",
                depends_on=["cached", "new"],
            )

            plan = TaskPlan(
                nodes={"cached": cached_node, "new": new_node, "root": root},
                root_key="root",
            )

            # Execute
            backend = LocalBackend(tmpdir, registry.get)

            completed_nodes = []

            def on_complete(key):
                completed_nodes.append(key)

            handle = backend.submit_plan(plan, on_node_complete=on_complete)

            # All nodes complete (cached node completes instantly via manifest check)
            assert "cached" in completed_nodes
            assert "new" in completed_nodes
            assert "root" in completed_nodes
            assert backend.get_plan_state(handle.plan_id) == "success"

            # Verify the cached node's original result.json was not overwritten
            original_result = _json.loads((cached_dir / "result.json").read_text())
            assert original_result == {"id": "cached", "status": "completed"}

    def test_execute_plan_with_failure(self):
        """Should handle node failure and call failure callback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build plan with a failing node
            node = TaskNode(
                key="failing",
                function="test.failing_task",
                subject_key="test:1",
                kwargs={},
                storage_prefix=f"{tmpdir}/failing",
                depends_on=[],
            )
            plan = TaskPlan(nodes={"failing": node}, root_key="failing")

            # Execute
            backend = LocalBackend(tmpdir, registry.get)

            failed_nodes = []

            def on_failure(key, error):
                failed_nodes.append((key, error))

            with pytest.raises(RuntimeError):
                backend.submit_plan(plan, on_node_failure=on_failure)

            # Verify
            assert backend.get_plan_state(plan.root_key) == "failure"
            assert len(failed_nodes) == 1
            assert failed_nodes[0][0] == "failing"
            assert "Intentional failure" in failed_nodes[0][1]

    def test_execute_plan_stops_on_failure(self):
        """Should stop execution when a node fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build plan: failing_node -> dependent_node
            failing = TaskNode(
                key="failing",
                function="test.failing_task",
                subject_key="test:1",
                kwargs={},
                storage_prefix=f"{tmpdir}/failing",
                depends_on=[],
            )
            dependent = TaskNode(
                key="dependent",
                function="test.leaf_task",
                subject_key="test:2",
                kwargs={"id": "dependent"},
                storage_prefix=f"{tmpdir}/dependent",
                depends_on=["failing"],
            )
            plan = TaskPlan(
                nodes={"failing": failing, "dependent": dependent},
                root_key="dependent",
            )

            # Execute
            backend = LocalBackend(tmpdir, registry.get)

            with pytest.raises(RuntimeError):
                backend.submit_plan(plan)

            # Dependent node should not have been executed
            dependent_result = Path(tmpdir) / "dependent" / "result.json"
            assert not dependent_result.exists()

    def test_cancel_raises_not_implemented(self):
        """Should raise NotImplementedError for cancel."""
        backend = LocalBackend("/tmp", registry.get)

        with pytest.raises(NotImplementedError):
            backend.cancel_plan("some-plan-id")

    def test_get_state_returns_pending_for_unknown(self):
        """Should return 'pending' for unknown plan IDs."""
        backend = LocalBackend("/tmp", registry.get)

        assert backend.get_plan_state("unknown-id") == "pending"


class TestExecutionLevels:
    """Tests for parallel execution level computation."""

    def test_diamond_dependency(self):
        """Should handle diamond-shaped dependencies correctly.

        Plan structure:
            A
           / \\
          B   C
           \\ /
            D
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            a = TaskNode(
                key="A", function="test.leaf_task",
                subject_key="test:A", kwargs={"id": "A"},
                storage_prefix=f"{tmpdir}/A", depends_on=[],
            )
            b = TaskNode(
                key="B", function="test.leaf_task",
                subject_key="test:B", kwargs={"id": "B"},
                storage_prefix=f"{tmpdir}/B", depends_on=["A"],
            )
            c = TaskNode(
                key="C", function="test.leaf_task",
                subject_key="test:C", kwargs={"id": "C"},
                storage_prefix=f"{tmpdir}/C", depends_on=["A"],
            )
            d = TaskNode(
                key="D", function="test.dependent_task",
                subject_key="test:D", kwargs={},
                storage_prefix=f"{tmpdir}/D", depends_on=["B", "C"],
            )

            plan = TaskPlan(
                nodes={"A": a, "B": b, "C": c, "D": d},
                root_key="D",
            )

            # Execute
            backend = LocalBackend(tmpdir, registry.get)

            order = []

            def on_complete(key):
                order.append(key)

            backend.submit_plan(plan, on_node_complete=on_complete)

            # Verify order: A must be first, D must be last
            assert order[0] == "A"
            assert order[-1] == "D"
            # B and C can be in any order between A and D
            assert set(order[1:3]) == {"B", "C"}
