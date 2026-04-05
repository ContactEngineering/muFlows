"""Tests for the Pipeline abstraction."""

import pytest

from muflow import register_task
from muflow.pipeline import ForEach, Pipeline, Step
from muflow.registry import clear


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear registry before and after each test."""
    clear()
    yield
    clear()


def _register_noop(name):
    """Register a no-op task for testing."""

    @register_task(name=name)
    def noop(context):
        pass

    return noop


class TestPipelineBuildPlan:
    """Tests for Pipeline.build_plan()."""

    def test_single_step(self):
        """A single-step pipeline produces one node."""
        _register_noop("test.single")
        p = Pipeline(
            name="p.single",
            steps={"only": Step(task="test.single")},
        )
        plan = p.build_plan("tag:1", {"x": 1})

        assert len(plan.nodes) == 1
        node = plan.nodes[plan.root_key]
        assert node.function == "test.single"
        assert node.kwargs == {"x": 1}
        assert node.depends_on == []

    def test_two_sequential_steps(self):
        """Two steps with after wiring."""
        _register_noop("test.a")
        _register_noop("test.b")
        p = Pipeline(
            name="p.seq",
            steps={
                "first": Step(task="test.a"),
                "second": Step(task="test.b", after=["first"]),
            },
        )
        plan = p.build_plan("tag:1", {})

        assert len(plan.nodes) == 2
        a_node = next(n for n in plan.nodes.values() if n.function == "test.a")
        b_node = next(n for n in plan.nodes.values() if n.function == "test.b")
        assert a_node.key in b_node.depends_on
        assert b_node.key in a_node.depended_on_by
        assert plan.root_key == b_node.key

    def test_foreach_fanout(self):
        """ForEach produces one node per item."""
        _register_noop("test.worker")
        p = Pipeline(
            name="p.fan",
            steps={
                "work": ForEach(
                    task="test.worker",
                    over=lambda sk, kw: [{"i": j} for j in range(3)],
                ),
            },
        )
        plan = p.build_plan("tag:1", {})

        # 3 workers + 1 sentinel root
        assert len(plan.nodes) == 4
        workers = [n for n in plan.nodes.values() if n.function == "test.worker"]
        assert len(workers) == 3

    def test_foreach_then_step(self):
        """A Step after a ForEach depends on all ForEach nodes."""
        _register_noop("test.worker")
        _register_noop("test.aggregate")
        p = Pipeline(
            name="p.fan_in",
            steps={
                "work": ForEach(
                    task="test.worker",
                    over=lambda sk, kw: [{"i": j} for j in range(3)],
                ),
                "agg": Step(task="test.aggregate", after=["work"]),
            },
        )
        plan = p.build_plan("tag:1", {})

        agg_node = next(
            n for n in plan.nodes.values() if n.function == "test.aggregate"
        )
        worker_keys = {
            n.key for n in plan.nodes.values() if n.function == "test.worker"
        }
        assert worker_keys == set(agg_node.depends_on)
        assert plan.root_key == agg_node.key

    def test_dependency_access_map_single_step(self):
        """A Step after a single Step gets plain step name as access key."""
        _register_noop("test.a")
        _register_noop("test.b")
        p = Pipeline(
            name="p.dep",
            steps={
                "first": Step(task="test.a"),
                "second": Step(task="test.b", after=["first"]),
            },
        )
        plan = p.build_plan("tag:1", {})

        b_node = next(n for n in plan.nodes.values() if n.function == "test.b")
        assert "first" in b_node.dependency_access_map
        a_node = next(n for n in plan.nodes.values() if n.function == "test.a")
        assert b_node.dependency_access_map["first"] == a_node.storage_prefix

    def test_dependency_access_map_foreach(self):
        """A Step after a ForEach gets colon-indexed access keys."""
        _register_noop("test.worker")
        _register_noop("test.agg")
        p = Pipeline(
            name="p.dep_fan",
            steps={
                "work": ForEach(
                    task="test.worker",
                    over=lambda sk, kw: [{"i": j} for j in range(3)],
                ),
                "agg": Step(task="test.agg", after=["work"]),
            },
        )
        plan = p.build_plan("tag:1", {})

        agg_node = next(n for n in plan.nodes.values() if n.function == "test.agg")
        assert "work:0" in agg_node.dependency_access_map
        assert "work:1" in agg_node.dependency_access_map
        assert "work:2" in agg_node.dependency_access_map

    def test_kwargs_merging(self):
        """Per-job kwargs are merged with pipeline kwargs."""
        _register_noop("test.worker")
        p = Pipeline(
            name="p.merge",
            steps={
                "work": ForEach(
                    task="test.worker",
                    over=lambda sk, kw: [{"extra": "val"}],
                ),
            },
        )
        plan = p.build_plan("tag:1", {"base": "param"})

        node = next(n for n in plan.nodes.values() if n.function == "test.worker")
        assert node.kwargs == {"base": "param", "extra": "val"}



class TestPipelineValidation:
    """Tests for pipeline validation."""

    def test_colon_in_step_name_raises(self):
        """Step names must not contain colons."""
        with pytest.raises(ValueError, match="must not contain ':'"):
            p = Pipeline(
                name="p.bad",
                steps={"bad:name": Step(task="test.x")},
            )
            p.build_plan("tag:1", {})

    def test_unknown_after_reference_raises(self):
        """Referencing a nonexistent step raises ValueError."""
        _register_noop("test.a")
        p = Pipeline(
            name="p.bad_ref",
            steps={
                "only": Step(task="test.a", after=["nonexistent"]),
            },
        )
        with pytest.raises(ValueError, match="unknown step"):
            p.build_plan("tag:1", {})

    def test_circular_after_raises(self):
        """Circular after references raise ValueError."""
        _register_noop("test.a")
        _register_noop("test.b")
        p = Pipeline(
            name="p.cycle",
            steps={
                "a": Step(task="test.a", after=["b"]),
                "b": Step(task="test.b", after=["a"]),
            },
        )
        with pytest.raises(ValueError, match="Circular dependency"):
            p.build_plan("tag:1", {})


class TestPipelineReadyNodesProgression:
    """Tests for walking through a multi-stage pipeline."""

    def test_three_stage_progression(self):
        """fan-out → fan-in → single step progression."""
        _register_noop("test.worker")
        _register_noop("test.aggregate")
        _register_noop("test.report")
        p = Pipeline(
            name="p.stages",
            steps={
                "work": ForEach(
                    task="test.worker",
                    over=lambda sk, kw: [{"i": j} for j in range(3)],
                ),
                "agg": Step(task="test.aggregate", after=["work"]),
                "report": Step(task="test.report", after=["agg"]),
            },
        )
        plan = p.build_plan("tag:1", {})
        completed = set()

        # Round 1: 3 workers
        ready = plan.ready_nodes(completed)
        assert len(ready) == 3
        assert all(n.function == "test.worker" for n in ready)
        completed.update(n.key for n in ready)

        # Round 2: aggregate
        ready = plan.ready_nodes(completed)
        assert len(ready) == 1
        assert ready[0].function == "test.aggregate"
        completed.add(ready[0].key)

        # Round 3: report
        ready = plan.ready_nodes(completed)
        assert len(ready) == 1
        assert ready[0].function == "test.report"
        completed.add(ready[0].key)

        assert plan.is_complete(completed)


class TestPipelineSentinel:
    """Tests for sentinel root node creation."""

    def test_sentinel_created_for_multi_terminal(self):
        """A sentinel root is created when the last step has multiple nodes."""
        _register_noop("test.worker")
        p = Pipeline(
            name="p.sentinel",
            steps={
                "work": ForEach(
                    task="test.worker",
                    over=lambda sk, kw: [{"i": j} for j in range(3)],
                ),
            },
        )
        plan = p.build_plan("tag:1", {})

        root = plan.nodes[plan.root_key]
        assert root.function == "p.sentinel"
        assert len(root.depends_on) == 3

    def test_no_sentinel_for_single_terminal(self):
        """No sentinel if the last step has exactly one node."""
        _register_noop("test.single")
        p = Pipeline(
            name="p.no_sentinel",
            steps={"only": Step(task="test.single")},
        )
        plan = p.build_plan("tag:1", {})

        root = plan.nodes[plan.root_key]
        assert root.function == "test.single"
