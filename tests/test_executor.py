"""Tests for the pure execution function."""

import json
import tempfile
from pathlib import Path

import pydantic

from muflow import TaskEntry, create_local_context
from muflow.executor import ExecutionPayload, ExecutionResult, execute_task


def mock_task_fn(ctx):
    user_id = ctx.kwargs.user_id if ctx.kwargs else 0
    ctx.save_json("result.json", {"status": "ok", "user_id": user_id})


class MockParams(pydantic.BaseModel):
    user_id: int = 0


def failing_task_fn(ctx):
    raise ValueError("Intentional failure for testing")


# Simple registry for tests
TEST_REGISTRY = {
    "test.mock_task": TaskEntry(
        name="test.mock_task",
        fn=mock_task_fn,
        parameters=MockParams,
    ),
    "test.failing_task": TaskEntry(
        name="test.failing_task",
        fn=failing_task_fn,
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
            task_name="test.task",
            kwargs={"param": "value"},
            storage_prefix="results/test",
        )
        assert payload.task_name == "test.task"
        assert payload.kwargs == {"param": "value"}
        assert payload.storage_prefix == "results/test"
        assert payload.dependency_prefixes == {}

    def test_to_dict(self):
        """Should serialize to dictionary."""
        payload = ExecutionPayload(
            task_name="test.task",
            kwargs={"param": "value"},
            storage_prefix="results/test",
            dependency_prefixes={"dep1": "results/dep1"},
        )
        d = payload.to_dict()

        assert d["task_name"] == "test.task"
        assert d["kwargs"] == {"param": "value"}
        assert d["storage_prefix"] == "results/test"
        assert d["dependency_prefixes"] == {"dep1": "results/dep1"}

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        d = {
            "task_name": "test.task",
            "kwargs": {"param": "value"},
            "storage_prefix": "results/test",
            "dependency_prefixes": {"dep1": "results/dep1"},
        }
        payload = ExecutionPayload.from_dict(d)

        assert payload.task_name == "test.task"
        assert payload.kwargs == {"param": "value"}
        assert payload.storage_prefix == "results/test"
        assert payload.dependency_prefixes == {"dep1": "results/dep1"}

    def test_roundtrip(self):
        """Should survive serialization roundtrip."""
        original = ExecutionPayload(
            task_name="test.task",
            kwargs={"param": "value"},
            storage_prefix="results/test",
            dependency_prefixes={"dep1": "results/dep1"},
        )
        restored = ExecutionPayload.from_dict(original.to_dict())

        assert restored.task_name == original.task_name
        assert restored.kwargs == original.kwargs
        assert restored.storage_prefix == original.storage_prefix
        assert restored.dependency_prefixes == original.dependency_prefixes


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_success_result(self):
        """Should create success result."""
        result = ExecutionResult(success=True)
        assert result.success is True
        assert result.error_message is None
        assert result.error_traceback is None

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
        result = ExecutionResult(success=True)
        d = result.to_dict()

        assert d["success"] is True
        assert d["error_message"] is None

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        d = {
            "success": False,
            "error_message": "Error",
            "error_traceback": "Traceback",
        }
        result = ExecutionResult.from_dict(d)

        assert result.success is False
        assert result.error_message == "Error"
        assert result.error_traceback == "Traceback"


class TestExecuteTask:
    """Tests for execute_task function."""

    def test_successful_execution(self):
        """Should execute task and return success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                task_name="test.mock_task",
                kwargs={"user_id": 42},
                storage_prefix=tmpdir,
            )

            ctx = create_local_context(path=tmpdir, kwargs=payload.kwargs)

            result = execute_task(payload, ctx, get_test_implementation)

            assert result.success is True
            assert result.error_message is None

            # Verify file was written
            data = ctx.read_json("result.json")
            assert data["status"] == "ok"
            assert data["user_id"] == 42

    def test_successful_execution_writes_manifest(self):
        """Should write manifest.json after execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                task_name="test.mock_task",
                kwargs={"user_id": 1},
                storage_prefix=tmpdir,
            )
            ctx = create_local_context(path=tmpdir, kwargs=payload.kwargs)

            execute_task(payload, ctx, get_test_implementation)

            manifest_path = Path(tmpdir) / "manifest.json"
            assert manifest_path.exists()
            manifest = json.loads(manifest_path.read_text())
            assert "result.json" in manifest["files"]

    def test_failed_execution(self):
        """Should catch exceptions and return failure result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                task_name="test.failing_task",
                kwargs={},
                storage_prefix=tmpdir,
            )

            ctx = create_local_context(path=tmpdir, kwargs={})

            result = execute_task(payload, ctx, get_test_implementation)

            assert result.success is False
            assert "Intentional failure" in result.error_message
            assert result.error_traceback is not None
            assert "ValueError" in result.error_traceback

    def test_failed_execution_still_writes_manifest(self):
        """Should write manifest.json even on failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                task_name="test.failing_task",
                kwargs={},
                storage_prefix=tmpdir,
            )
            ctx = create_local_context(path=tmpdir, kwargs={})

            execute_task(payload, ctx, get_test_implementation)

            manifest_path = Path(tmpdir) / "manifest.json"
            assert manifest_path.exists()

    def test_unknown_task(self):
        """Should return failure for unknown task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                task_name="nonexistent.task",
                kwargs={},
                storage_prefix=tmpdir,
            )

            ctx = create_local_context(path=tmpdir, kwargs={})

            result = execute_task(payload, ctx, get_test_implementation)

            assert result.success is False
            assert "nonexistent.task" in result.error_message

    def test_protected_file_write_fails(self):
        """Task that writes to a protected file should fail."""

        def bad_task_fn(ctx):
            ctx.save_json("context.json", {})

        registry = {
            "test.bad_task": TaskEntry(
                name="test.bad_task",
                fn=bad_task_fn,
            )
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                task_name="test.bad_task",
                kwargs={},
                storage_prefix=tmpdir,
            )

            ctx = create_local_context(path=tmpdir, kwargs={})

            result = execute_task(payload, ctx, lambda name: registry[name])

            assert result.success is False
            assert "protected" in result.error_message

    def test_cached_execution_skips_task(self):
        """Should skip task execution when manifest.json already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create a manifest to simulate a cached result
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text('{"files": ["result.json"], "timestamp": "2024-01-01T00:00:00+00:00"}')

            payload = ExecutionPayload(
                task_name="test.mock_task",
                kwargs={"user_id": 99},
                storage_prefix=tmpdir,
            )
            ctx = create_local_context(path=tmpdir, kwargs=payload.kwargs)

            result = execute_task(payload, ctx, get_test_implementation)

            assert result.success is True
            assert result.cached is True

            # Manifest should remain unchanged (not overwritten by execute_task)
            assert manifest_path.read_text() == '{"files": ["result.json"], "timestamp": "2024-01-01T00:00:00+00:00"}'

    def test_normal_execution_has_cached_false(self):
        """Normal (non-cached) execution should have cached=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                task_name="test.mock_task",
                kwargs={"user_id": 1},
                storage_prefix=tmpdir,
            )
            ctx = create_local_context(path=tmpdir, kwargs=payload.kwargs)

            result = execute_task(payload, ctx, get_test_implementation)

            assert result.success is True
            assert result.cached is False

    def test_executor_writes_context_json(self):
        """Should write context.json before execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                task_name="test.mock_task",
                kwargs={"user_id": 1},
                storage_prefix=tmpdir,
                context_data={"foo": "bar"}
            )
            ctx = create_local_context(path=tmpdir, kwargs=payload.kwargs)

            execute_task(payload, ctx, get_test_implementation)

            context_path = Path(tmpdir) / "context.json"
            assert context_path.exists()
            context_data = json.loads(context_path.read_text())
            assert context_data == {"foo": "bar"}
