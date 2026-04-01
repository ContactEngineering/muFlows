"""Tests for the pure execution function."""

import json
import tempfile
from pathlib import Path

import pydantic

from muflow import WorkflowEntry, create_local_context
from muflow.executor import ExecutionPayload, ExecutionResult, execute_workflow


def mock_workflow_fn(ctx):
    user_id = ctx.kwargs.user_id if ctx.kwargs else 0
    ctx.save_json("result.json", {"status": "ok", "user_id": user_id})


class MockParams(pydantic.BaseModel):
    user_id: int = 0


def failing_workflow_fn(ctx):
    raise ValueError("Intentional failure for testing")


# Simple registry for tests
TEST_REGISTRY = {
    "test.mock_workflow": WorkflowEntry(
        name="test.mock_workflow",
        fn=mock_workflow_fn,
        parameters=MockParams,
    ),
    "test.failing_workflow": WorkflowEntry(
        name="test.failing_workflow",
        fn=failing_workflow_fn,
    ),
}


def get_test_implementation(name):
    """Test implementation getter."""
    return TEST_REGISTRY[name]


class TestExecutionPayload:
    """Tests for ExecutionPayload dataclass."""

    def test_creation(self):
        """Should create payload with required fields."""
        payload = ExecutionPayload(
            workflow_name="test.workflow",
            kwargs={"param": "value"},
            storage_prefix="results/test",
        )
        assert payload.workflow_name == "test.workflow"
        assert payload.kwargs == {"param": "value"}
        assert payload.storage_prefix == "results/test"
        assert payload.dependency_prefixes == {}

    def test_to_dict(self):
        """Should serialize to dictionary."""
        payload = ExecutionPayload(
            workflow_name="test.workflow",
            kwargs={"param": "value"},
            storage_prefix="results/test",
            dependency_prefixes={"dep1": "results/dep1"},
        )
        d = payload.to_dict()

        assert d["workflow_name"] == "test.workflow"
        assert d["kwargs"] == {"param": "value"}
        assert d["storage_prefix"] == "results/test"
        assert d["dependency_prefixes"] == {"dep1": "results/dep1"}

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        d = {
            "workflow_name": "test.workflow",
            "kwargs": {"param": "value"},
            "storage_prefix": "results/test",
            "dependency_prefixes": {"dep1": "results/dep1"},
        }
        payload = ExecutionPayload.from_dict(d)

        assert payload.workflow_name == "test.workflow"
        assert payload.kwargs == {"param": "value"}
        assert payload.storage_prefix == "results/test"
        assert payload.dependency_prefixes == {"dep1": "results/dep1"}

    def test_roundtrip(self):
        """Should survive serialization roundtrip."""
        original = ExecutionPayload(
            workflow_name="test.workflow",
            kwargs={"param": "value"},
            storage_prefix="results/test",
            dependency_prefixes={"dep1": "results/dep1"},
        )
        restored = ExecutionPayload.from_dict(original.to_dict())

        assert restored.workflow_name == original.workflow_name
        assert restored.kwargs == original.kwargs
        assert restored.storage_prefix == original.storage_prefix
        assert restored.dependency_prefixes == original.dependency_prefixes


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_success_result(self):
        """Should create success result."""
        result = ExecutionResult(
            success=True,
            files_written=["result.json"],
        )
        assert result.success is True
        assert result.error_message is None
        assert result.error_traceback is None
        assert result.files_written == ["result.json"]

    def test_failure_result(self):
        """Should create failure result."""
        result = ExecutionResult(
            success=False,
            error_message="Something went wrong",
            error_traceback="Traceback...",
        )
        assert result.success is False
        assert result.error_message == "Something went wrong"
        assert result.error_traceback == "Traceback..."

    def test_to_dict(self):
        """Should serialize to dictionary."""
        result = ExecutionResult(
            success=True,
            files_written=["result.json"],
        )
        d = result.to_dict()

        assert d["success"] is True
        assert d["error_message"] is None
        assert d["files_written"] == ["result.json"]

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        d = {
            "success": False,
            "error_message": "Error",
            "error_traceback": "Traceback",
            "files_written": [],
        }
        result = ExecutionResult.from_dict(d)

        assert result.success is False
        assert result.error_message == "Error"
        assert result.error_traceback == "Traceback"


class TestExecuteWorkflow:
    """Tests for execute_workflow function."""

    def test_successful_execution(self):
        """Should execute workflow and return success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                workflow_name="test.mock_workflow",
                kwargs={"user_id": 42},
                storage_prefix=tmpdir,
            )

            ctx = create_local_context(path=tmpdir, kwargs=payload.kwargs)

            result = execute_workflow(payload, ctx, get_test_implementation)

            assert result.success is True
            assert result.error_message is None
            assert "result.json" in result.files_written

            # Verify file was written
            data = ctx.read_json("result.json")
            assert data["status"] == "ok"
            assert data["user_id"] == 42

    def test_successful_execution_writes_manifest(self):
        """Should write manifest.json after execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                workflow_name="test.mock_workflow",
                kwargs={"user_id": 1},
                storage_prefix=tmpdir,
            )
            ctx = create_local_context(path=tmpdir, kwargs=payload.kwargs)

            execute_workflow(payload, ctx, get_test_implementation)

            manifest_path = Path(tmpdir) / "manifest.json"
            assert manifest_path.exists()
            manifest = json.loads(manifest_path.read_text())
            assert "result.json" in manifest["files"]

    def test_failed_execution(self):
        """Should catch exceptions and return failure result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                workflow_name="test.failing_workflow",
                kwargs={},
                storage_prefix=tmpdir,
            )

            ctx = create_local_context(path=tmpdir, kwargs={})

            result = execute_workflow(payload, ctx, get_test_implementation)

            assert result.success is False
            assert "Intentional failure" in result.error_message
            assert result.error_traceback is not None
            assert "ValueError" in result.error_traceback

    def test_failed_execution_still_writes_manifest(self):
        """Should write manifest.json even on failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                workflow_name="test.failing_workflow",
                kwargs={},
                storage_prefix=tmpdir,
            )
            ctx = create_local_context(path=tmpdir, kwargs={})

            execute_workflow(payload, ctx, get_test_implementation)

            manifest_path = Path(tmpdir) / "manifest.json"
            assert manifest_path.exists()

    def test_unknown_workflow(self):
        """Should return failure for unknown workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                workflow_name="nonexistent.workflow",
                kwargs={},
                storage_prefix=tmpdir,
            )

            ctx = create_local_context(path=tmpdir, kwargs={})

            result = execute_workflow(payload, ctx, get_test_implementation)

            assert result.success is False
            assert "nonexistent.workflow" in result.error_message

    def test_protected_file_write_fails(self):
        """Workflow that writes to a protected file should fail."""

        def bad_workflow_fn(ctx):
            ctx.save_json("context.json", {})

        registry = {
            "test.bad_workflow": WorkflowEntry(
                name="test.bad_workflow",
                fn=bad_workflow_fn,
            )
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                workflow_name="test.bad_workflow",
                kwargs={},
                storage_prefix=tmpdir,
            )

            ctx = create_local_context(path=tmpdir, kwargs={})

            result = execute_workflow(payload, ctx, lambda name: registry[name])

            assert result.success is False
            assert "protected" in result.error_message

    def test_executor_writes_context_json(self):
        """Should write context.json before execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                workflow_name="test.mock_workflow",
                kwargs={"user_id": 1},
                storage_prefix=tmpdir,
                context_data={"foo": "bar"}
            )
            ctx = create_local_context(path=tmpdir, kwargs=payload.kwargs)

            execute_workflow(payload, ctx, get_test_implementation)

            context_path = Path(tmpdir) / "context.json"
            assert context_path.exists()
            context_data = json.loads(context_path.read_text())
            assert context_data == {"foo": "bar"}
