"""Tests for WorkflowContext."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from muflow import WorkflowContext, create_local_context
from muflow.context import WorkflowContextProtocol


class TestWorkflowContext:
    """Tests for WorkflowContext."""

    def test_implements_protocol(self):
        """WorkflowContext should implement WorkflowContextProtocol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = create_local_context(path=tmpdir, kwargs={})
            assert isinstance(ctx, WorkflowContextProtocol)
            assert isinstance(ctx, WorkflowContext)

    def test_storage_prefix(self):
        """storage_prefix should return the path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = create_local_context(path=tmpdir, kwargs={})
            assert ctx.storage_prefix == tmpdir

    def test_kwargs(self):
        """kwargs should return the raw dict by default (not yet validated)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = {"param1": "value1", "param2": 42}
            ctx = create_local_context(path=tmpdir, kwargs=kwargs)
            assert ctx.kwargs == kwargs

    def test_save_and_read_json(self):
        """Should save and read JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = create_local_context(path=tmpdir, kwargs={})

            data = {"key": "value", "number": 42, "nested": {"a": 1}}
            ctx.save_json("test.json", data)

            assert ctx.exists("test.json")
            loaded = ctx.read_json("test.json")
            assert loaded == data

    def test_save_and_read_json_with_nan(self):
        """Should handle NaN values in JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = create_local_context(path=tmpdir, kwargs={})

            data = {"value": float("nan"), "inf": float("inf")}
            ctx.save_json("test.json", data)

            loaded = ctx.read_json("test.json")
            assert np.isnan(loaded["value"])
            assert np.isinf(loaded["inf"])

    def test_save_and_read_file(self):
        """Should save and read raw bytes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = create_local_context(path=tmpdir, kwargs={})

            data = b"Hello, World!"
            ctx.save_file("test.txt", data)

            assert ctx.exists("test.txt")
            loaded = ctx.read_file("test.txt")
            assert loaded == data

    def test_save_and_read_xarray(self):
        """Should save and read xarray Datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = create_local_context(path=tmpdir, kwargs={})

            ds = xr.Dataset({
                "temperature": (["x", "y"], np.random.rand(3, 4)),
                "pressure": (["x", "y"], np.random.rand(3, 4)),
            })
            ctx.save_xarray("test.nc", ds)

            assert ctx.exists("test.nc")
            loaded = ctx.read_xarray("test.nc")
            assert "temperature" in loaded
            assert "pressure" in loaded
            np.testing.assert_array_almost_equal(
                loaded["temperature"].values,
                ds["temperature"].values,
            )

    def test_open_file(self):
        """Should open files for reading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = create_local_context(path=tmpdir, kwargs={})

            ctx.save_json("test.json", {"key": "value"})

            with ctx.open_file("test.json", "r") as f:
                content = f.read()
                assert "key" in content
                assert "value" in content

    def test_exists_false_for_missing(self):
        """exists() should return False for missing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = create_local_context(path=tmpdir, kwargs={})
            assert not ctx.exists("nonexistent.json")

    def test_nested_directories(self):
        """Should handle nested directory paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = create_local_context(path=tmpdir, kwargs={})

            ctx.save_json("subdir/nested/test.json", {"key": "value"})
            assert ctx.exists("subdir/nested/test.json")
            loaded = ctx.read_json("subdir/nested/test.json")
            assert loaded == {"key": "value"}

    def test_dependency_access(self):
        """Should access dependency outputs via dependency()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dependency output
            dep_path = Path(tmpdir) / "dependency"
            dep_path.mkdir()
            dep_ctx = create_local_context(path=str(dep_path), kwargs={})
            dep_ctx.save_json("result.json", {"dep_value": 123})

            # Create main context with dependency
            main_path = Path(tmpdir) / "main"
            main_ctx = create_local_context(
                path=str(main_path),
                kwargs={},
                dependency_paths={"dep1": str(dep_path)},
            )

            # Access dependency
            dep = main_ctx.dependency("dep1")
            result = dep.read_json("result.json")
            assert result == {"dep_value": 123}

    def test_dependency_unknown_raises(self):
        """dependency() should raise KeyError for unknown dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = create_local_context(path=tmpdir, kwargs={})

            with pytest.raises(KeyError):
                ctx.dependency("unknown")

    def test_creates_directory_if_missing(self):
        """Should create the directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "new_dir"
            assert not path.exists()

            create_local_context(path=str(path), kwargs={})
            assert path.exists()


class TestStorageSafetyViaContext:
    """Tests for storage backend safety features accessed through the context."""

    def test_write_once_via_context(self):
        """Context delegates write-once enforcement to the storage backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = create_local_context(path=tmpdir, kwargs={})
            ctx.save_json("data.json", {"v": 1})
            with pytest.raises(FileExistsError):
                ctx.save_json("data.json", {"v": 2})

    def test_protected_files_via_context(self):
        """Context delegates protected-file enforcement to the storage backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = create_local_context(path=tmpdir, kwargs={})
            with pytest.raises(PermissionError, match="protected"):
                ctx.save_json("context.json", {})
            with pytest.raises(PermissionError, match="protected"):
                ctx.save_json("manifest.json", {})

    def test_path_traversal_via_context(self):
        """Context delegates path traversal protection to the storage backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = create_local_context(path=tmpdir, kwargs={})
            with pytest.raises(ValueError, match="traversal"):
                ctx.save_json("../escape.json", {})

    def test_dependency_reading_works(self):
        """Dependency contexts can read existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dep_path = Path(tmpdir) / "dependency"
            dep_path.mkdir()
            dep_ctx = create_local_context(path=str(dep_path), kwargs={})
            dep_ctx.save_json("result.json", {"dep_value": 123})

            main_path = Path(tmpdir) / "main"
            main_ctx = create_local_context(
                path=str(main_path),
                kwargs={},
                dependency_paths={"dep1": str(dep_path)},
            )

            dep = main_ctx.dependency("dep1")
            result = dep.read_json("result.json")
            assert result == {"dep_value": 123}

    def test_storage_property(self):
        """Context exposes the storage backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = create_local_context(path=tmpdir, kwargs={})
            assert ctx.storage is not None
            ctx.save_json("test.json", {})
            assert "test.json" in ctx.storage.written_files
