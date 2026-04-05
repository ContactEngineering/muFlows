"""Tests for TaskPlan and related functions."""

import json

import pytest

from muflow import TaskNode, TaskPlan, compute_prefix


class TestComputePrefix:
    """Tests for compute_prefix()."""

    def test_deterministic(self):
        """Same inputs should produce same prefix."""
        hash_dict = {
            "task": "my.task",
            "subject": "subject:123",
            "param": "value",
        }
        prefix1 = compute_prefix(hash_dict)
        prefix2 = compute_prefix(hash_dict)
        assert prefix1 == prefix2

    def test_different_function_different_prefix(self):
        """Different function names should produce different prefixes."""
        prefix1 = compute_prefix({"task": "task.a", "subject": "subject:123"})
        prefix2 = compute_prefix({"task": "task.b", "subject": "subject:123"})
        assert prefix1 != prefix2

    def test_different_subject_different_prefix(self):
        """Different subjects should produce different prefixes."""
        prefix1 = compute_prefix({"task": "my.task", "subject": "subject:123"})
        prefix2 = compute_prefix({"task": "my.task", "subject": "subject:456"})
        assert prefix1 != prefix2

    def test_different_kwargs_different_prefix(self):
        """Different kwargs should produce different prefixes."""
        prefix1 = compute_prefix(
            {"task": "my.task", "subject": "subject:123", "a": 1}
        )
        prefix2 = compute_prefix(
            {"task": "my.task", "subject": "subject:123", "a": 2}
        )
        assert prefix1 != prefix2

    def test_kwargs_order_independent(self):
        """Kwargs order should not affect prefix."""
        prefix1 = compute_prefix(
            {"task": "my.task", "subject": "subject:123", "a": 1, "b": 2}
        )
        prefix2 = compute_prefix(
            {"task": "my.task", "subject": "subject:123", "b": 2, "a": 1}
        )
        assert prefix1 == prefix2

    def test_includes_function_name(self):
        """Prefix should include function name for readability."""
        prefix = compute_prefix(
            {"task": "sds_ml.v3.gpr.training", "subject": "tag:1"}
        )
        assert "sds_ml.v3.gpr.training" in prefix

    def test_custom_base_prefix(self):
        """Should support custom base prefix."""
        prefix = compute_prefix(
            {"task": "my.task", "subject": "subject:123"},
            base_prefix="custom/prefix",
        )
        assert prefix.startswith("custom/prefix/")


class TestTaskNode:
    """Tests for TaskNode dataclass."""

    def test_creation(self):
        """Should create node with required fields."""
        node = TaskNode(
            key="test/node",
            function="my.task",
            subject_key="subject:123",
            kwargs={"param": "value"},
            storage_prefix="data-lake/results/my.task/abc123",
        )
        assert node.key == "test/node"
        assert node.function == "my.task"
        assert node.depends_on == []

    def test_to_dict(self):
        """Should serialize to dict."""
        node = TaskNode(
            key="test/node",
            function="my.task",
            subject_key="subject:123",
            kwargs={"param": "value"},
            storage_prefix="prefix",
            output_files=["result.json"],
        )
        d = node.to_dict()
        assert d["key"] == "test/node"
        assert d["function"] == "my.task"
        assert d["output_files"] == ["result.json"]

    def test_from_dict(self):
        """Should deserialize from dict."""
        d = {
            "key": "test/node",
            "function": "my.task",
            "subject_key": "subject:123",
            "kwargs": {"param": "value"},
            "storage_prefix": "prefix",
            "depends_on": ["dep1"],
            "depended_on_by": [],
            "output_files": ["result.json"],
            "analysis_id": 42,
        }
        node = TaskNode.from_dict(d)
        assert node.key == "test/node"
        assert node.depends_on == ["dep1"]
        assert node.analysis_id == 42


class TestTaskPlan:
    """Tests for TaskPlan."""

    @pytest.fixture
    def simple_plan(self):
        """Create a simple plan with 3 nodes: A -> B -> C."""
        nodes = {
            "A": TaskNode(
                key="A",
                function="task.a",
                subject_key="s1",
                kwargs={},
                storage_prefix="prefix/a",
                depends_on=[],
                depended_on_by=["B"],
            ),
            "B": TaskNode(
                key="B",
                function="task.b",
                subject_key="s1",
                kwargs={},
                storage_prefix="prefix/b",
                depends_on=["A"],
                depended_on_by=["C"],
            ),
            "C": TaskNode(
                key="C",
                function="task.c",
                subject_key="s1",
                kwargs={},
                storage_prefix="prefix/c",
                depends_on=["B"],
                depended_on_by=[],
            ),
        }
        return TaskPlan(nodes=nodes, root_key="C")

    @pytest.fixture
    def fan_out_plan(self):
        """Create a plan with fan-out: A -> [B1, B2, B3] -> C."""
        nodes = {
            "A": TaskNode(
                key="A",
                function="task.a",
                subject_key="s1",
                kwargs={},
                storage_prefix="prefix/a",
                depends_on=[],
                depended_on_by=["B1", "B2", "B3"],
            ),
            "B1": TaskNode(
                key="B1",
                function="task.b",
                subject_key="s1",
                kwargs={"fold": 1},
                storage_prefix="prefix/b1",
                depends_on=["A"],
                depended_on_by=["C"],
            ),
            "B2": TaskNode(
                key="B2",
                function="task.b",
                subject_key="s1",
                kwargs={"fold": 2},
                storage_prefix="prefix/b2",
                depends_on=["A"],
                depended_on_by=["C"],
            ),
            "B3": TaskNode(
                key="B3",
                function="task.b",
                subject_key="s1",
                kwargs={"fold": 3},
                storage_prefix="prefix/b3",
                depends_on=["A"],
                depended_on_by=["C"],
            ),
            "C": TaskNode(
                key="C",
                function="task.c",
                subject_key="s1",
                kwargs={},
                storage_prefix="prefix/c",
                depends_on=["B1", "B2", "B3"],
                depended_on_by=[],
            ),
        }
        return TaskPlan(nodes=nodes, root_key="C")

    def test_leaf_nodes(self, simple_plan):
        """leaf_nodes() should return nodes with no dependencies."""
        leaves = simple_plan.leaf_nodes()
        assert len(leaves) == 1
        assert leaves[0].key == "A"

    def test_leaf_nodes_fan_out(self, fan_out_plan):
        """leaf_nodes() should return A in fan-out plan."""
        leaves = fan_out_plan.leaf_nodes()
        assert len(leaves) == 1
        assert leaves[0].key == "A"

    def test_ready_nodes_initial(self, simple_plan):
        """ready_nodes() should return A when nothing completed."""
        ready = simple_plan.ready_nodes(completed=set())
        assert len(ready) == 1
        assert ready[0].key == "A"

    def test_ready_nodes_after_a(self, simple_plan):
        """ready_nodes() should return B after A completes."""
        ready = simple_plan.ready_nodes(completed={"A"})
        assert len(ready) == 1
        assert ready[0].key == "B"

    def test_ready_nodes_after_b(self, simple_plan):
        """ready_nodes() should return C after A and B complete."""
        ready = simple_plan.ready_nodes(completed={"A", "B"})
        assert len(ready) == 1
        assert ready[0].key == "C"

    def test_ready_nodes_fan_out(self, fan_out_plan):
        """ready_nodes() should return B1, B2, B3 after A completes."""
        ready = fan_out_plan.ready_nodes(completed={"A"})
        keys = {n.key for n in ready}
        assert keys == {"B1", "B2", "B3"}

    def test_ready_nodes_fan_in(self, fan_out_plan):
        """ready_nodes() should return C only after all B's complete."""
        # After just B1 and B2
        ready = fan_out_plan.ready_nodes(completed={"A", "B1", "B2"})
        keys = {n.key for n in ready}
        assert "C" not in keys
        assert "B3" in keys

        # After all B's
        ready = fan_out_plan.ready_nodes(completed={"A", "B1", "B2", "B3"})
        keys = {n.key for n in ready}
        assert keys == {"C"}

    def test_is_complete(self, simple_plan):
        """is_complete() should return True when root is complete."""
        assert not simple_plan.is_complete(completed=set())
        assert not simple_plan.is_complete(completed={"A"})
        assert not simple_plan.is_complete(completed={"A", "B"})
        assert simple_plan.is_complete(completed={"A", "B", "C"})

    def test_to_dict(self, simple_plan):
        """Should serialize to dict."""
        d = simple_plan.to_dict()
        assert d["root_key"] == "C"
        assert "A" in d["nodes"]
        assert "B" in d["nodes"]
        assert "C" in d["nodes"]

    def test_from_dict(self, simple_plan):
        """Should deserialize from dict."""
        d = simple_plan.to_dict()
        plan = TaskPlan.from_dict(d)
        assert plan.root_key == "C"
        assert len(plan.nodes) == 3
        assert plan.nodes["A"].function == "task.a"

    def test_to_json_and_back(self, simple_plan):
        """Should round-trip through JSON."""
        json_str = simple_plan.to_json()
        plan = TaskPlan.from_json(json_str)
        assert plan.root_key == simple_plan.root_key
        assert len(plan.nodes) == len(simple_plan.nodes)

    def test_json_is_valid(self, simple_plan):
        """to_json() should produce valid JSON."""
        json_str = simple_plan.to_json()
        parsed = json.loads(json_str)
        assert "root_key" in parsed
        assert "nodes" in parsed
