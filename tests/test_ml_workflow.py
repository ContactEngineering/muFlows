"""Tests for the ML workflow pipeline example."""

import pytest

from muflow.registry import clear


DATASETS = ["dataset_a", "dataset_b", "dataset_c"]


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear registry before and after each test."""
    clear()
    # Import triggers @register_workflow decorators
    from muflow.examples import ml_workflow  # noqa: F401
    yield
    clear()


@pytest.fixture
def plan():
    """Build a plan for the ML pipeline with 3 datasets."""
    from muflow.examples.ml_workflow import ml_pipeline

    return ml_pipeline.build_plan(
        subject_key="experiment:1",
        kwargs={"datasets": DATASETS},
    )


class TestMLWorkflowDAGStructure:
    """Tests for the DAG topology."""

    def test_node_count(self, plan):
        """3 features + 1 train + 3 loo_cv + 3 reports + 1 sentinel = 11."""
        assert len(plan.nodes) == 11

    def test_workflow_counts(self, plan):
        """Correct number of each workflow type."""
        functions = [n.function for n in plan.nodes.values()]
        assert functions.count("ml.compute_features") == 3
        assert functions.count("ml.train_model") == 1
        assert functions.count("ml.loo_cv") == 3
        assert functions.count("ml.generate_report") == 3
        assert functions.count("ml.full_pipeline") == 1  # sentinel

    def test_leaf_nodes_are_features(self, plan):
        """The only leaf nodes should be the 3 feature computations."""
        leaves = plan.leaf_nodes()
        assert len(leaves) == 3
        assert all(n.function == "ml.compute_features" for n in leaves)

    def test_feature_nodes_have_correct_datasets(self, plan):
        """Each feature node gets a different dataset_name."""
        feature_nodes = [
            n for n in plan.nodes.values()
            if n.function == "ml.compute_features"
        ]
        dataset_names = {n.kwargs["dataset_name"] for n in feature_nodes}
        assert dataset_names == set(DATASETS)

    def test_train_depends_on_all_features(self, plan):
        """Train model depends on all 3 feature nodes."""
        feature_keys = {
            n.key for n in plan.nodes.values()
            if n.function == "ml.compute_features"
        }
        train_node = next(
            n for n in plan.nodes.values() if n.function == "ml.train_model"
        )
        assert feature_keys == set(train_node.depends_on)

    def test_loo_cv_depends_on_all_features(self, plan):
        """Each LOO CV node depends on all 3 feature nodes."""
        feature_keys = {
            n.key for n in plan.nodes.values()
            if n.function == "ml.compute_features"
        }
        cv_nodes = [
            n for n in plan.nodes.values() if n.function == "ml.loo_cv"
        ]
        assert len(cv_nodes) == 3
        for cv_node in cv_nodes:
            assert feature_keys == set(cv_node.depends_on)

    def test_reports_depend_on_train_and_all_cv(self, plan):
        """Each report depends on train_model + all LOO CV nodes."""
        train_key = next(
            n.key for n in plan.nodes.values()
            if n.function == "ml.train_model"
        )
        cv_keys = {
            n.key for n in plan.nodes.values() if n.function == "ml.loo_cv"
        }
        report_nodes = [
            n for n in plan.nodes.values()
            if n.function == "ml.generate_report"
        ]
        assert len(report_nodes) == 3
        for rn in report_nodes:
            assert train_key in rn.depends_on
            assert cv_keys.issubset(set(rn.depends_on))

    def test_sentinel_depends_on_all_reports(self, plan):
        """The sentinel root depends on all 3 report nodes."""
        sentinel = plan.nodes[plan.root_key]
        assert sentinel.function == "ml.full_pipeline"
        report_keys = {
            n.key for n in plan.nodes.values()
            if n.function == "ml.generate_report"
        }
        assert report_keys == set(sentinel.depends_on)


class TestMLWorkflowDependencyAccess:
    """Tests for dependency_access_map correctness."""

    def test_train_model_access_map(self, plan):
        """Train model has features:0, features:1, features:2 access keys."""
        train_node = next(
            n for n in plan.nodes.values() if n.function == "ml.train_model"
        )
        assert set(train_node.dependency_access_map.keys()) == {
            "features:0", "features:1", "features:2",
        }

    def test_loo_cv_access_map(self, plan):
        """Each LOO CV has features:0, features:1, features:2 access keys."""
        cv_nodes = [
            n for n in plan.nodes.values() if n.function == "ml.loo_cv"
        ]
        for cv_node in cv_nodes:
            assert set(cv_node.dependency_access_map.keys()) == {
                "features:0", "features:1", "features:2",
            }

    def test_report_access_map(self, plan):
        """Each report has train + loo_cv:0, loo_cv:1, loo_cv:2 access keys."""
        report_nodes = [
            n for n in plan.nodes.values()
            if n.function == "ml.generate_report"
        ]
        for rn in report_nodes:
            assert "train" in rn.dependency_access_map
            assert "loo_cv:0" in rn.dependency_access_map
            assert "loo_cv:1" in rn.dependency_access_map
            assert "loo_cv:2" in rn.dependency_access_map


class TestMLWorkflowReadyNodesProgression:
    """Tests for walking through the DAG execution order."""

    def test_full_progression(self, plan):
        """Walk through all 4 stages + sentinel."""
        completed = set()

        # Round 1: 3 feature nodes
        ready = plan.ready_nodes(completed)
        assert len(ready) == 3
        assert all(n.function == "ml.compute_features" for n in ready)
        completed.update(n.key for n in ready)

        # Round 2: 1 train + 3 loo_cv
        ready = plan.ready_nodes(completed)
        assert len(ready) == 4
        functions = {n.function for n in ready}
        assert functions == {"ml.train_model", "ml.loo_cv"}
        completed.update(n.key for n in ready)

        # Round 3: 3 reports
        ready = plan.ready_nodes(completed)
        assert len(ready) == 3
        assert all(n.function == "ml.generate_report" for n in ready)
        completed.update(n.key for n in ready)

        # Round 4: sentinel
        ready = plan.ready_nodes(completed)
        assert len(ready) == 1
        assert ready[0].function == "ml.full_pipeline"
        completed.add(ready[0].key)

        assert plan.is_complete(completed)


class TestMLWorkflowCaching:
    """Tests for caching in the ML pipeline."""

    def test_cached_features_skip_to_training(self):
        """If all features are cached, training becomes immediately ready."""
        from muflow.examples.ml_workflow import ml_pipeline

        plan = ml_pipeline.build_plan(
            subject_key="experiment:1",
            kwargs={"datasets": DATASETS},
            is_cached=lambda name, *_: name == "ml.compute_features",
        )

        ready = plan.ready_nodes(set())
        functions = {n.function for n in ready}
        assert "ml.compute_features" not in functions
        assert "ml.train_model" in functions
        assert "ml.loo_cv" in functions
