"""Tests for execution backends."""

import tempfile
from pathlib import Path

import pytest

from muflow import WorkflowNode, WorkflowPlan, register_workflow, registry
from muflow.backends import LocalBackend


# Test workflows
@register_workflow(name="test.leaf_workflow")
def leaf_workflow(context):
    """Simple leaf workflow that writes a result."""
    params = context.kwargs
    context.save_json("result.json", {
        "id": params.get("id", "unknown"),
        "status": "completed",
    })


@register_workflow(name="test.dependent_workflow")
def dependent_workflow(context):
    """Workflow that reads from dependencies."""
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


@register_workflow(name="test.failing_workflow")
def failing_workflow(context):
    """Workflow that always fails."""
    raise ValueError("Intentional failure for testing")


class TestLocalBackend:
    """Tests for LocalBackend."""

    def test_execute_single_node_plan(self):
        """Should execute a plan with a single node."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build a simple plan with one node
            node = WorkflowNode(
                key="node1",
                function="test.leaf_workflow",
                subject_key="test:1",
                kwargs={"id": "node1"},
                storage_prefix=f"{tmpdir}/node1",
                depends_on=[],
            )
            plan = WorkflowPlan(nodes={"node1": node}, root_key="node1")

            # Execute
            backend = LocalBackend(tmpdir, registry.get)
            plan_id = backend.submit_plan(plan)

            # Verify
            assert backend.get_plan_state(plan_id) == "success"
            result_path = Path(tmpdir) / "node1" / "result.json"
            assert result_path.exists()

    def test_execute_plan_with_dependencies(self):
        """Should execute nodes in dependency order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build a plan: 3 leaf nodes -> 1 dependent node
            nodes = {}

            # Leaf nodes
            for i in range(3):
                node = WorkflowNode(
                    key=f"leaf_{i}",
                    function="test.leaf_workflow",
                    subject_key=f"test:{i}",
                    kwargs={"id": f"leaf_{i}"},
                    storage_prefix=f"{tmpdir}/leaf_{i}",
                    depends_on=[],
                )
                nodes[node.key] = node

            # Dependent node
            root = WorkflowNode(
                key="root",
                function="test.dependent_workflow",
                subject_key="test:root",
                kwargs={},
                storage_prefix=f"{tmpdir}/root",
                depends_on=["leaf_0", "leaf_1", "leaf_2"],
            )
            nodes["root"] = root

            plan = WorkflowPlan(nodes=nodes, root_key="root")

            # Execute
            backend = LocalBackend(tmpdir, registry.get)

            completed_nodes = []

            def on_complete(key):
                completed_nodes.append(key)

            plan_id = backend.submit_plan(plan, on_node_complete=on_complete)

            # Verify execution order
            assert backend.get_plan_state(plan_id) == "success"
            assert len(completed_nodes) == 4

            # Leaf nodes should complete before root
            root_idx = completed_nodes.index("root")
            for i in range(3):
                leaf_idx = completed_nodes.index(f"leaf_{i}")
                assert leaf_idx < root_idx, "Leaf should complete before root"

    def test_execute_plan_with_cached_nodes(self):
        """Should skip cached nodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a pre-existing result for one node
            cached_dir = Path(tmpdir) / "cached"
            cached_dir.mkdir()
            (cached_dir / "result.json").write_text(
                '{"id": "cached", "status": "completed"}'
            )

            # Build plan with one cached node
            cached_node = WorkflowNode(
                key="cached",
                function="test.leaf_workflow",
                subject_key="test:cached",
                kwargs={"id": "cached"},
                storage_prefix=str(cached_dir),
                depends_on=[],
                cached=True,  # Mark as cached
            )
            new_node = WorkflowNode(
                key="new",
                function="test.leaf_workflow",
                subject_key="test:new",
                kwargs={"id": "new"},
                storage_prefix=f"{tmpdir}/new",
                depends_on=[],
            )
            root = WorkflowNode(
                key="root",
                function="test.dependent_workflow",
                subject_key="test:root",
                kwargs={},
                storage_prefix=f"{tmpdir}/root",
                depends_on=["cached", "new"],
            )

            plan = WorkflowPlan(
                nodes={"cached": cached_node, "new": new_node, "root": root},
                root_key="root",
            )

            # Execute
            backend = LocalBackend(tmpdir, registry.get)

            executed_nodes = []

            def on_complete(key):
                executed_nodes.append(key)

            plan_id = backend.submit_plan(plan, on_node_complete=on_complete)

            # Verify cached node was skipped
            assert "cached" not in executed_nodes
            assert "new" in executed_nodes
            assert "root" in executed_nodes
            assert backend.get_plan_state(plan_id) == "success"

    def test_execute_plan_with_failure(self):
        """Should handle node failure and call failure callback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build plan with a failing node
            node = WorkflowNode(
                key="failing",
                function="test.failing_workflow",
                subject_key="test:1",
                kwargs={},
                storage_prefix=f"{tmpdir}/failing",
                depends_on=[],
            )
            plan = WorkflowPlan(nodes={"failing": node}, root_key="failing")

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
            failing = WorkflowNode(
                key="failing",
                function="test.failing_workflow",
                subject_key="test:1",
                kwargs={},
                storage_prefix=f"{tmpdir}/failing",
                depends_on=[],
            )
            dependent = WorkflowNode(
                key="dependent",
                function="test.leaf_workflow",
                subject_key="test:2",
                kwargs={"id": "dependent"},
                storage_prefix=f"{tmpdir}/dependent",
                depends_on=["failing"],
            )
            plan = WorkflowPlan(
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
            a = WorkflowNode(
                key="A", function="test.leaf_workflow",
                subject_key="test:A", kwargs={"id": "A"},
                storage_prefix=f"{tmpdir}/A", depends_on=[],
            )
            b = WorkflowNode(
                key="B", function="test.leaf_workflow",
                subject_key="test:B", kwargs={"id": "B"},
                storage_prefix=f"{tmpdir}/B", depends_on=["A"],
            )
            c = WorkflowNode(
                key="C", function="test.leaf_workflow",
                subject_key="test:C", kwargs={"id": "C"},
                storage_prefix=f"{tmpdir}/C", depends_on=["A"],
            )
            d = WorkflowNode(
                key="D", function="test.dependent_workflow",
                subject_key="test:D", kwargs={},
                storage_prefix=f"{tmpdir}/D", depends_on=["B", "C"],
            )

            plan = WorkflowPlan(
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
