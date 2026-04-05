"""Tests for CeleryBackend."""

from unittest.mock import MagicMock, patch

import pytest

from muflow.plan import TaskPlan

celery_pkg = pytest.importorskip("celery", reason="celery required")

from celery import Celery  # noqa: E402

from muflow.backends.celery import CeleryBackend, create_celery_task  # noqa: E402

from tests.conftest import (  # noqa: E402
    diamond_plan,
    fan_in_plan,
    linear_plan,
    simple_plan,
)

try:
    import boto3
    from moto import mock_aws

    HAS_S3_DEPS = True
except ImportError:
    HAS_S3_DEPS = False


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def celery_app():
    """In-process Celery app with eager execution (no broker needed)."""
    app = Celery("muflow-test")
    app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
        broker_url="memory://",
        result_backend="cache+memory://",
    )
    return app


@pytest.fixture
def backend(celery_app):
    return CeleryBackend(
        celery_app, bucket="test-bucket", base_prefix="muflow"
    )


@pytest.fixture(params=["celery", "step_functions"])
def levels_backend(request, celery_app):
    """Parametrized fixture: same _compute_levels tests run on both backends."""
    if request.param == "celery":
        yield CeleryBackend(celery_app, bucket="test-bucket")
    else:
        boto3_mod = pytest.importorskip("boto3")
        pytest.importorskip("moto")
        from moto import mock_aws as _mock_aws

        from muflow.backends.step_functions import StepFunctionsBackend

        with _mock_aws():
            client = boto3_mod.client(
                "stepfunctions", region_name="us-east-1"
            )
            yield StepFunctionsBackend(
                function_arn="arn:aws:lambda:us-east-1:123:function:w",
                bucket="test-bucket",
                role_arn="arn:aws:iam::123:role/r",
                sfn_client=client,
            )


# ── TestComputeLevels (parameterized: Celery + StepFunctions) ────────────────


class TestComputeLevels:
    def test_single_node(self, levels_backend):
        plan = simple_plan()
        levels = levels_backend._compute_levels(plan)
        assert len(levels) == 1
        assert levels[0][0].key == "muflow/test.simple/aaa"

    def test_linear_two_nodes(self, levels_backend):
        plan = linear_plan()
        levels = levels_backend._compute_levels(plan)
        assert len(levels) == 2
        assert levels[0][0].function == "test.dep"
        assert levels[1][0].function == "test.root"

    def test_fan_in(self, levels_backend):
        plan = fan_in_plan()
        levels = levels_backend._compute_levels(plan)
        assert len(levels) == 2
        assert len(levels[0]) == 3  # three leaves in parallel
        assert len(levels[1]) == 1  # one root

    def test_diamond(self, levels_backend):
        plan = diamond_plan()
        levels = levels_backend._compute_levels(plan)
        assert len(levels) == 3
        assert levels[0][0].function == "test.a"
        assert len(levels[1]) == 2  # B and C in parallel
        assert levels[2][0].function == "test.d"


# ── TestBuildCeleryTask ──────────────────────────────────────────────────


class TestBuildCeleryTask:
    def test_single_level_returns_group(self, backend):
        plan = fan_in_plan()
        levels = backend._compute_levels(plan)
        # Only take level 0 (3 leaves)
        task = backend._build_celery_task([levels[0]], plan)
        from celery.canvas import group

        assert isinstance(task, group)

    def test_single_node_level_returns_group(self, backend):
        plan = simple_plan()
        levels = backend._compute_levels(plan)
        task = backend._build_celery_task(levels, plan)
        from celery.canvas import group

        assert isinstance(task, group)

    def test_two_levels_returns_chord(self, backend):
        plan = linear_plan()
        levels = backend._compute_levels(plan)
        task = backend._build_celery_task(levels, plan)
        from celery.canvas import chord

        assert isinstance(task, chord)

    def test_three_levels_is_nested_chord(self, backend):
        plan = diamond_plan()
        levels = backend._compute_levels(plan)
        task = backend._build_celery_task(levels, plan)
        from celery.canvas import chord

        assert isinstance(task, chord)


# ── TestMakeNodeTask ─────────────────────────────────────────────────────────


class TestMakeNodeTask:
    def test_signature_task_name(self, backend):
        plan = simple_plan()
        node = plan.nodes["muflow/test.simple/aaa"]
        sig = backend._make_node_task(node, plan)
        assert sig.task == "muflow.execute_node"

    def test_signature_args_node_key(self, backend):
        plan = simple_plan()
        node = plan.nodes["muflow/test.simple/aaa"]
        sig = backend._make_node_task(node, plan)
        assert sig.args[0] == "muflow/test.simple/aaa"

    def test_payload_dict_keys(self, backend):
        plan = simple_plan()
        node = plan.nodes["muflow/test.simple/aaa"]
        sig = backend._make_node_task(node, plan)
        payload_dict = sig.args[1]
        assert "task_name" in payload_dict
        assert "kwargs" in payload_dict
        assert "storage_prefix" in payload_dict
        assert "dependency_prefixes" in payload_dict

    def test_payload_dependency_prefixes(self, backend):
        plan = linear_plan()
        root = plan.nodes["muflow/test.root/ccc"]
        # Set up dependency access map
        root.dependency_access_map = {"dep": "muflow/test.dep/bbb"}
        sig = backend._make_node_task(root, plan)
        payload_dict = sig.args[1]
        assert payload_dict["dependency_prefixes"] == {
            "dep": "muflow/test.dep/bbb"
        }

    def test_default_queue(self, backend):
        plan = simple_plan()
        node = plan.nodes["muflow/test.simple/aaa"]
        sig = backend._make_node_task(node, plan)
        assert sig.options.get("queue") == "default"

    def test_custom_queue(self, backend):
        plan = simple_plan()
        node = plan.nodes["muflow/test.simple/aaa"]
        # Simulate a node-like object that has a queue attribute
        mock_node = MagicMock(wraps=node)
        mock_node.queue = "gpu"
        mock_node.function = node.function
        mock_node.kwargs = node.kwargs
        mock_node.storage_prefix = node.storage_prefix
        mock_node.key = node.key
        mock_node.dependency_access_map = node.dependency_access_map
        sig = backend._make_node_task(mock_node, plan)
        assert sig.options.get("queue") == "gpu"


# ── TestGetPlanState ─────────────────────────────────────────────────────────


class TestGetPlanState:
    def test_pending(self, backend):
        mock_result = MagicMock()
        mock_result.state = "PENDING"
        backend._plan_results["test-id"] = mock_result
        assert backend.get_plan_state("test-id") == "pending"

    def test_started(self, backend):
        mock_result = MagicMock()
        mock_result.state = "STARTED"
        backend._plan_results["test-id"] = mock_result
        assert backend.get_plan_state("test-id") == "running"

    def test_success(self, backend):
        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        backend._plan_results["test-id"] = mock_result
        assert backend.get_plan_state("test-id") == "success"

    def test_failure(self, backend):
        mock_result = MagicMock()
        mock_result.state = "FAILURE"
        backend._plan_results["test-id"] = mock_result
        assert backend.get_plan_state("test-id") == "failure"

    def test_revoked(self, backend):
        mock_result = MagicMock()
        mock_result.state = "REVOKED"
        backend._plan_results["test-id"] = mock_result
        assert backend.get_plan_state("test-id") == "failure"

    def test_unknown_state_maps_to_pending(self, backend):
        mock_result = MagicMock()
        mock_result.state = "WEIRD_UNKNOWN"
        backend._plan_results["test-id"] = mock_result
        assert backend.get_plan_state("test-id") == "pending"

    def test_looks_up_unknown_plan_id_from_celery(self, backend):
        """plan_id not in _plan_results → falls back to AsyncResult."""
        with patch("celery.result.AsyncResult") as mock_cls:
            mock_result = MagicMock()
            mock_result.state = "SUCCESS"
            mock_cls.return_value = mock_result
            assert backend.get_plan_state("unknown-id") == "success"


# ── TestCancelPlan ───────────────────────────────────────────────────────────


class TestCancelPlan:
    def test_revoke_called_with_terminate(self, backend):
        backend._app.control = MagicMock()
        backend.cancel_plan("test-plan-id")
        backend._app.control.revoke.assert_called_once_with(
            "test-plan-id", terminate=True
        )


# ── TestSubmitPlanEager (end-to-end with eager + moto S3) ────────────────────


@pytest.mark.skipif(not HAS_S3_DEPS, reason="boto3 and moto required")
class TestSubmitPlanEager:
    @pytest.fixture
    def s3_env(self, celery_app, clean_registry):
        """Set up moto S3, registry, and eager backend."""
        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="test-bucket")

            # Use a mutable registry shared via the Celery task closure.
            # We clear it between tests to avoid cross-contamination.
            # Use a unique task name so re-registration works.
            task_name = f"muflow.execute_node.eager.{id(s3)}"
            test_registry = {}
            create_celery_task(
                celery_app,
                task_registry=test_registry,
                task_name=task_name,
            )
            backend = CeleryBackend(
                celery_app,
                bucket="test-bucket",
                base_prefix="muflow",
                task_name=task_name,
            )
            yield backend, test_registry, s3

    def _register_noop(self, registry, name="test.simple"):
        from muflow.registry import TaskEntry

        def noop(ctx):
            ctx.save_json("output.json", {"status": "ok"})

        entry = TaskEntry(name=name, fn=noop)
        registry[name] = entry

    def _register_failing(self, registry, name="test.fail"):
        from muflow.registry import TaskEntry

        def fail(ctx):
            raise RuntimeError("intentional failure")

        entry = TaskEntry(name=name, fn=fail)
        registry[name] = entry

    def test_single_node_returns_plan_handle(self, s3_env):
        from muflow.backends.handle import PlanHandle

        backend, registry, s3 = s3_env
        self._register_noop(registry)
        plan = simple_plan()
        handle = backend.submit_plan(plan)
        assert isinstance(handle, PlanHandle)
        assert handle.backend == "celery"
        assert len(handle.plan_id) > 0

    def test_state_is_success_after_eager_submit(self, s3_env):
        backend, registry, s3 = s3_env
        self._register_noop(registry)
        plan = simple_plan()
        handle = backend.submit_plan(plan)
        assert backend.get_plan_state(handle.plan_id) == "success"

    def test_task_output_written_to_s3(self, s3_env):
        backend, registry, s3 = s3_env
        self._register_noop(registry)
        plan = simple_plan()
        backend.submit_plan(plan)

        # Check that the task wrote output.json to S3
        prefix = "muflow/test.simple/aaa"
        resp = s3.list_objects_v2(Bucket="test-bucket", Prefix=prefix)
        keys = [obj["Key"] for obj in resp.get("Contents", [])]
        assert any("output.json" in k for k in keys)

    def test_fan_in_plan_executes_all_nodes(self, s3_env):
        backend, registry, s3 = s3_env
        self._register_noop(registry, name="test.leaf")
        self._register_noop(registry, name="test.root")
        plan = fan_in_plan()
        handle = backend.submit_plan(plan)
        assert backend.get_plan_state(handle.plan_id) == "success"

        # Verify all 4 nodes wrote output
        resp = s3.list_objects_v2(Bucket="test-bucket", Prefix="muflow/")
        keys = [obj["Key"] for obj in resp.get("Contents", [])]
        output_keys = [k for k in keys if "output.json" in k]
        assert len(output_keys) == 4

    def test_failed_task_raises(self, s3_env):
        backend, registry, s3 = s3_env
        self._register_failing(registry, name="test.simple")
        plan = simple_plan()
        with pytest.raises(RuntimeError, match="intentional failure"):
            backend.submit_plan(plan)


# ── TestCreateCeleryTask ─────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_S3_DEPS, reason="boto3 and moto required")
class TestCreateCeleryTask:
    @pytest.fixture
    def task_env(self, celery_app, clean_registry):
        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="test-bucket")

            test_registry = {}
            task = create_celery_task(
                celery_app,
                task_registry=test_registry,
                task_name="muflow.execute_node",
            )
            yield task, test_registry, s3

    def test_task_registered_with_name(self, celery_app):
        create_celery_task(
            celery_app,
            task_registry={},
            task_name="muflow.execute_node",
        )
        assert "muflow.execute_node" in celery_app.tasks

    def test_custom_task_name(self, celery_app):
        create_celery_task(
            celery_app,
            task_registry={},
            task_name="custom.task.name",
        )
        assert "custom.task.name" in celery_app.tasks

    def test_unknown_task_raises_value_error(self, task_env):
        task, registry, s3 = task_env
        payload_dict = {
            "task_name": "nonexistent.task",
            "kwargs": {},
            "storage_prefix": "muflow/test/aaa",
            "dependency_prefixes": {},
        }
        with pytest.raises(ValueError, match="Unknown task"):
            task("node-key", payload_dict, "test-bucket")

    def test_known_task_executes(self, task_env):
        task, registry, s3 = task_env
        from muflow.registry import TaskEntry

        def noop(ctx):
            ctx.save_json("result.json", {"done": True})

        registry["test.wf"] = TaskEntry(name="test.wf", fn=noop)

        payload_dict = {
            "task_name": "test.wf",
            "kwargs": {},
            "storage_prefix": "muflow/test.wf/aaa",
            "dependency_prefixes": {},
        }
        result = task("node-key", payload_dict, "test-bucket")
        assert result["node_key"] == "node-key"

    def test_execution_result_json_written(self, task_env):
        task, registry, s3 = task_env
        from muflow.registry import TaskEntry

        def noop(ctx):
            ctx.save_json("out.json", {})

        registry["test.wf"] = TaskEntry(name="test.wf", fn=noop)

        payload_dict = {
            "task_name": "test.wf",
            "kwargs": {},
            "storage_prefix": "muflow/test.wf/bbb",
            "dependency_prefixes": {},
        }
        task("node-key", payload_dict, "test-bucket")

        resp = s3.list_objects_v2(
            Bucket="test-bucket", Prefix="muflow/test.wf/bbb"
        )
        keys = [obj["Key"] for obj in resp.get("Contents", [])]
        assert any("_execution_result.json" in k for k in keys)

    def test_failed_task_raises_runtime_error(self, task_env):
        task, registry, s3 = task_env
        from muflow.registry import TaskEntry

        def fail(ctx):
            raise RuntimeError("boom")

        registry["test.fail"] = TaskEntry(name="test.fail", fn=fail)

        payload_dict = {
            "task_name": "test.fail",
            "kwargs": {},
            "storage_prefix": "muflow/test.fail/ccc",
            "dependency_prefixes": {},
        }
        with pytest.raises(RuntimeError, match="boom"):
            task("node-key", payload_dict, "test-bucket")
