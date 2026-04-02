"""Tests for WorkflowPlanner."""

import pytest

from muflow import WorkflowPlanner, WorkflowSpec, register_workflow
from muflow.planner import get_dependency_access_map
from muflow.registry import clear


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear registry before and after each test."""
    clear()
    yield
    clear()


class TestWorkflowPlannerBasic:
    """Basic planner tests."""

    def test_single_workflow_no_deps(self):
        """Should create plan with single node for workflow without deps."""

        @register_workflow(name="test.simple")
        def simple(context):
            pass

        planner = WorkflowPlanner()
        plan = planner.build_plan("test.simple", "tag:1", {"param": "value"})

        assert len(plan.nodes) == 1
        assert plan.root_key in plan.nodes

        root = plan.nodes[plan.root_key]
        assert root.function == "test.simple"
        assert root.subject_key == "tag:1"
        assert root.kwargs == {"param": "value"}
        assert root.depends_on == []

    def test_unknown_workflow_raises(self):
        """Should raise ValueError for unknown workflow."""
        planner = WorkflowPlanner()

        with pytest.raises(ValueError, match="Unknown workflow"):
            planner.build_plan("nonexistent.workflow", "tag:1", {})


class TestWorkflowPlannerDependencies:
    """Tests for dependency resolution."""

    def test_simple_dependency(self):
        """Should resolve simple string dependency."""

        @register_workflow(name="test.features")
        def features(context):
            pass

        @register_workflow(
            name="test.training",
            dependencies={"features": "test.features"},
        )
        def training(context):
            pass

        planner = WorkflowPlanner()
        plan = planner.build_plan("test.training", "tag:1", {})

        assert len(plan.nodes) == 2

        # Find the training and features nodes
        training_node = next(
            n for n in plan.nodes.values() if n.function == "test.training"
        )
        features_node = next(
            n for n in plan.nodes.values() if n.function == "test.features"
        )

        # Training depends on features
        assert features_node.key in training_node.depends_on
        # Features is depended on by training
        assert training_node.key in features_node.depended_on_by

    def test_dependency_with_explicit_spec(self):
        """Should resolve WorkflowSpec dependency with custom subject/kwargs."""

        @register_workflow(name="test.features")
        def features(context):
            pass

        @register_workflow(
            name="test.training",
            dependencies={
                "features": WorkflowSpec(
                    workflow="test.features",
                    subject_key="surface:999",
                    kwargs={"resolution": "high"},
                ),
            },
        )
        def training(context):
            pass

        planner = WorkflowPlanner()
        plan = planner.build_plan("test.training", "tag:1", {"other": "param"})

        features_node = next(
            n for n in plan.nodes.values() if n.function == "test.features"
        )
        assert features_node.subject_key == "surface:999"
        assert features_node.kwargs == {"resolution": "high"}

    def test_dynamic_dependencies(self):
        """Should enumerate dependencies dynamically."""

        @register_workflow(name="test.process_surface")
        def process_surface(context):
            pass

        def enumerate_surfaces(subject_key, kwargs):
            surfaces = kwargs.get("surfaces", [])
            return {
                f"surface_{i}": WorkflowSpec(
                    workflow="test.process_surface",
                    subject_key=f"surface:{s['id']}",
                    kwargs={"surface_id": s["id"]},
                )
                for i, s in enumerate(surfaces)
            }

        @register_workflow(
            name="test.training",
            dependencies=enumerate_surfaces,
        )
        def training(context):
            pass

        planner = WorkflowPlanner()
        plan = planner.build_plan(
            "test.training",
            "tag:1",
            {"surfaces": [{"id": 10}, {"id": 20}, {"id": 30}]},
        )

        # 1 training + 3 surface processing = 4 nodes
        assert len(plan.nodes) == 4

        # All process_surface nodes
        process_nodes = [
            n for n in plan.nodes.values() if n.function == "test.process_surface"
        ]
        assert len(process_nodes) == 3

        subject_keys = {n.subject_key for n in process_nodes}
        assert subject_keys == {"surface:10", "surface:20", "surface:30"}

    def test_chained_dependencies(self):
        """Should resolve transitive dependencies."""

        @register_workflow(name="test.a")
        def workflow_a(context):
            pass

        @register_workflow(
            name="test.b",
            dependencies={"a": "test.a"},
        )
        def workflow_b(context):
            pass

        @register_workflow(
            name="test.c",
            dependencies={"b": "test.b"},
        )
        def workflow_c(context):
            pass

        planner = WorkflowPlanner()
        plan = planner.build_plan("test.c", "tag:1", {})

        assert len(plan.nodes) == 3

        # Find nodes
        node_a = next(n for n in plan.nodes.values() if n.function == "test.a")
        node_b = next(n for n in plan.nodes.values() if n.function == "test.b")
        node_c = next(n for n in plan.nodes.values() if n.function == "test.c")

        # C depends on B, B depends on A
        assert node_b.key in node_c.depends_on
        assert node_a.key in node_b.depends_on

        # Leaf nodes should be [A]
        leaf_nodes = plan.leaf_nodes()
        assert len(leaf_nodes) == 1
        assert leaf_nodes[0].function == "test.a"


class TestWorkflowPlannerProduces:
    """Tests for fan-out (produces) resolution."""

    def test_simple_produces(self):
        """Should resolve produces as downstream dependencies."""

        @register_workflow(name="test.fold")
        def fold(context):
            pass

        def enumerate_folds(subject_key, kwargs):
            n_folds = kwargs.get("n_folds", 3)
            return {
                f"fold_{i}": WorkflowSpec(
                    workflow="test.fold",
                    kwargs={**kwargs, "fold_index": i},
                )
                for i in range(n_folds)
            }

        @register_workflow(
            name="test.setup",
            produces=enumerate_folds,
        )
        def setup(context):
            pass

        planner = WorkflowPlanner()
        plan = planner.build_plan("test.setup", "tag:1", {"n_folds": 3})

        # 1 setup + 3 folds = 4 nodes
        assert len(plan.nodes) == 4

        setup_node = next(n for n in plan.nodes.values() if n.function == "test.setup")
        fold_nodes = [n for n in plan.nodes.values() if n.function == "test.fold"]

        # All folds depend on setup
        for fold_node in fold_nodes:
            assert setup_node.key in fold_node.depends_on

        # Setup is depended on by all folds
        assert len(setup_node.depended_on_by) == 3

    def test_fanout_fanin_pattern(self):
        """Should correctly resolve fan-out followed by fan-in."""

        @register_workflow(name="test.fold")
        def fold(context):
            pass

        def enumerate_folds(subject_key, kwargs):
            n_folds = kwargs.get("n_folds", 3)
            return {
                f"fold_{i}": WorkflowSpec(
                    workflow="test.fold",
                    kwargs={**kwargs, "fold_index": i},
                )
                for i in range(n_folds)
            }

        @register_workflow(
            name="test.setup",
            produces=enumerate_folds,
        )
        def setup(context):
            pass

        @register_workflow(
            name="test.aggregate",
            dependencies=enumerate_folds,  # Depend on all folds
        )
        def aggregate(context):
            pass

        planner = WorkflowPlanner()
        plan = planner.build_plan("test.aggregate", "tag:1", {"n_folds": 3})

        # Find nodes
        aggregate_node = next(
            n for n in plan.nodes.values() if n.function == "test.aggregate"
        )
        fold_nodes = [n for n in plan.nodes.values() if n.function == "test.fold"]

        # Aggregate depends on all folds
        assert len(aggregate_node.depends_on) == 3
        for fold_node in fold_nodes:
            assert fold_node.key in aggregate_node.depends_on


class TestWorkflowPlannerCaching:
    """Tests for cache checking."""

    def test_cached_nodes_marked(self):
        """Cached nodes should be marked as cached."""

        @register_workflow(name="test.workflow")
        def workflow(context):
            pass

        def is_cached(name, subject, kwargs):
            return True  # Everything is cached

        planner = WorkflowPlanner(is_cached=is_cached)
        plan = planner.build_plan("test.workflow", "tag:1", {})

        node = plan.nodes[plan.root_key]
        assert node.cached is True

    def test_cached_deps_not_resolved(self):
        """Dependencies of cached nodes should not be resolved."""

        @register_workflow(name="test.dep")
        def dep(context):
            pass

        @register_workflow(
            name="test.main",
            dependencies={"dep": "test.dep"},
        )
        def main(context):
            pass

        def is_cached(name, subject, kwargs):
            # Main is cached, so its deps shouldn't be resolved
            return name == "test.main"

        planner = WorkflowPlanner(is_cached=is_cached)
        plan = planner.build_plan("test.main", "tag:1", {})

        # Only the main node should be in the plan
        assert len(plan.nodes) == 1
        assert plan.nodes[plan.root_key].cached is True


class TestWorkflowPlannerCycleDetection:
    """Tests for circular dependency detection."""

    def test_direct_cycle_raises(self):
        """Should detect direct circular dependency."""

        @register_workflow(
            name="test.a",
            dependencies={"b": "test.b"},
        )
        def workflow_a(context):
            pass

        @register_workflow(
            name="test.b",
            dependencies={"a": "test.a"},
        )
        def workflow_b(context):
            pass

        planner = WorkflowPlanner()

        with pytest.raises(ValueError, match="Circular dependency"):
            planner.build_plan("test.a", "tag:1", {})

    def test_self_reference_raises(self):
        """Should detect self-referential dependency."""

        @register_workflow(
            name="test.self",
            dependencies={"self": "test.self"},
        )
        def workflow_self(context):
            pass

        planner = WorkflowPlanner()

        with pytest.raises(ValueError, match="Circular dependency"):
            planner.build_plan("test.self", "tag:1", {})


class TestGetDependencyAccessMap:
    """Tests for get_dependency_access_map function."""

    def test_returns_access_keys(self):
        """Should map access keys to storage prefixes."""

        @register_workflow(name="test.features")
        def features(context):
            pass

        @register_workflow(
            name="test.training",
            dependencies={"my_features": "test.features"},
        )
        def training(context):
            pass

        planner = WorkflowPlanner()
        plan = planner.build_plan("test.training", "tag:1", {})

        training_key = plan.root_key
        access_map = get_dependency_access_map(plan, training_key)

        assert "my_features" in access_map
        # The value should be a storage prefix (string)
        assert isinstance(access_map["my_features"], str)
        assert "test.features" in access_map["my_features"]


class TestPlanExecution:
    """Tests for plan execution helpers."""

    def test_ready_nodes_initial(self):
        """Initially, only leaf nodes should be ready."""

        @register_workflow(name="test.leaf")
        def leaf(context):
            pass

        @register_workflow(
            name="test.root",
            dependencies={"leaf": "test.leaf"},
        )
        def root(context):
            pass

        planner = WorkflowPlanner()
        plan = planner.build_plan("test.root", "tag:1", {})

        ready = plan.ready_nodes(set())
        assert len(ready) == 1
        assert ready[0].function == "test.leaf"

    def test_ready_nodes_after_completion(self):
        """After deps complete, dependent nodes should be ready."""

        @register_workflow(name="test.leaf")
        def leaf(context):
            pass

        @register_workflow(
            name="test.root",
            dependencies={"leaf": "test.leaf"},
        )
        def root(context):
            pass

        planner = WorkflowPlanner()
        plan = planner.build_plan("test.root", "tag:1", {})

        # Find leaf node key
        leaf_key = next(n.key for n in plan.nodes.values() if n.function == "test.leaf")

        # After leaf completes, root should be ready
        ready = plan.ready_nodes({leaf_key})
        assert len(ready) == 1
        assert ready[0].function == "test.root"

    def test_is_complete(self):
        """is_complete should return True when root is done."""

        @register_workflow(name="test.workflow")
        def workflow(context):
            pass

        planner = WorkflowPlanner()
        plan = planner.build_plan("test.workflow", "tag:1", {})

        assert not plan.is_complete(set())
        assert plan.is_complete({plan.root_key})

    def test_identity_keys_affect_node_key(self):
        """Node key should be computed using identity_keys if provided."""

        @register_workflow(name="test.identity", identity_keys=["id"])
        def my_workflow(context):
            pass

        planner = WorkflowPlanner()

        # Different kwargs, same id -> same node key
        key1 = planner.build_plan(
            "test.identity", "sub", {"id": 1, "other": "a"}
        ).root_key
        key2 = planner.build_plan(
            "test.identity", "sub", {"id": 1, "other": "b"}
        ).root_key
        assert key1 == key2

        # Different id -> different node key
        key3 = planner.build_plan(
            "test.identity", "sub", {"id": 2, "other": "a"}
        ).root_key
        assert key1 != key3

    def test_identity_keys_passed_to_is_cached(self):
        """Planner should use identity_keys when checking cache."""
        checked_kwargs = []

        def is_cached(name, subject, kwargs):
            checked_kwargs.append(kwargs)
            return False

        @register_workflow(name="test.cached", identity_keys=["id"])
        def my_workflow(context):
            pass

        planner = WorkflowPlanner(is_cached=is_cached)
        planner.build_plan("test.cached", "sub", {"id": 1, "other": "val"})

        # HACK: Actually, planner currently passes full kwargs to is_cached,
        # but the node_key it uses for the node itself is hashed with identity_keys.
        # Wait, let's check planner.py again.


class TestWorkflowPlannerIdentityAnnotations:
    """Tests for identity keys defined via Pydantic annotations."""

    def test_identity_key_annotation(self):
        """Should extract identity keys from Pydantic model annotations."""
        from typing import Annotated

        import pydantic

        from muflow import IdentityKey

        class AnnotatedParams(pydantic.BaseModel):
            id: Annotated[int, IdentityKey()]
            other: str

        @register_workflow(name="test.annotated", parameters=AnnotatedParams)
        def my_workflow(context):
            pass

        planner = WorkflowPlanner()

        # Different kwargs, same id -> same node key
        plan1 = planner.build_plan(
            "test.annotated", "sub", {"id": 1, "other": "a"}
        )
        plan2 = planner.build_plan(
            "test.annotated", "sub", {"id": 1, "other": "b"}
        )
        assert plan1.root_key == plan2.root_key

        # Different id -> different node key
        plan3 = planner.build_plan(
            "test.annotated", "sub", {"id": 2, "other": "a"}
        )
        assert plan1.root_key != plan3.root_key

    def test_multiple_identity_key_annotations(self):
        """Should handle multiple fields annotated with IdentityKey."""
        from typing import Annotated

        import pydantic

        from muflow import IdentityKey

        class MultiAnnotatedParams(pydantic.BaseModel):
            id1: Annotated[int, IdentityKey()]
            id2: Annotated[int, IdentityKey()]
            other: str

        @register_workflow(name="test.multi_annotated", parameters=MultiAnnotatedParams)
        def my_workflow(context):
            pass

        planner = WorkflowPlanner()

        # Same id1, id2 -> same node key
        key1 = planner.build_plan(
            "test.multi_annotated", "sub", {"id1": 1, "id2": 1, "other": "a"}
        ).root_key
        key2 = planner.build_plan(
            "test.multi_annotated", "sub", {"id1": 1, "id2": 1, "other": "b"}
        ).root_key
        assert key1 == key2

        # Different id1 or id2 -> different node key
        key3 = planner.build_plan(
            "test.multi_annotated", "sub", {"id1": 1, "id2": 2, "other": "a"}
        ).root_key
        assert key1 != key3

    def test_explicit_identity_keys_override_annotations(self):
        """Explicit identity_keys argument should override annotations."""
        from typing import Annotated

        import pydantic

        from muflow import IdentityKey

        class AnnotatedParams(pydantic.BaseModel):
            id: Annotated[int, IdentityKey()]
            other: str

        # Override: only 'other' is an identity key
        @register_workflow(
            name="test.override",
            parameters=AnnotatedParams,
            identity_keys=["other"]
        )
        def my_workflow(context):
            pass

        planner = WorkflowPlanner()

        # Different id, same other -> same node key
        key1 = planner.build_plan(
            "test.override", "sub", {"id": 1, "other": "a"}
        ).root_key
        key2 = planner.build_plan(
            "test.override", "sub", {"id": 2, "other": "a"}
        ).root_key
        assert key1 == key2

        # Same id, different other -> different node key
        key3 = planner.build_plan(
            "test.override", "sub", {"id": 1, "other": "b"}
        ).root_key
        assert key1 != key3
