"""Tests for WorkflowContext implementations."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from muflow import LocalFolderContext, WorkflowContext


class TestLocalFolderContext:
    """Tests for LocalFolderContext."""

    def test_implements_protocol(self):
        """LocalFolderContext should implement WorkflowContext protocol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = LocalFolderContext(path=tmpdir, kwargs={})
            assert isinstance(ctx, WorkflowContext)

    def test_storage_prefix(self):
        """storage_prefix should return the path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = LocalFolderContext(path=tmpdir, kwargs={})
            assert ctx.storage_prefix == tmpdir

    def test_kwargs(self):
        """kwargs should return the provided parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = {"param1": "value1", "param2": 42}
            ctx = LocalFolderContext(path=tmpdir, kwargs=kwargs)
            assert ctx.kwargs == kwargs

    def test_save_and_read_json(self):
        """Should save and read JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = LocalFolderContext(path=tmpdir, kwargs={})

            data = {"key": "value", "number": 42, "nested": {"a": 1}}
            ctx.save_json("test.json", data)

            assert ctx.exists("test.json")
            loaded = ctx.read_json("test.json")
            assert loaded == data

    def test_save_and_read_json_with_nan(self):
        """Should handle NaN values in JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = LocalFolderContext(path=tmpdir, kwargs={})

            data = {"value": float("nan"), "inf": float("inf")}
            ctx.save_json("test.json", data)

            loaded = ctx.read_json("test.json")
            assert np.isnan(loaded["value"])
            assert np.isinf(loaded["inf"])

    def test_save_and_read_file(self):
        """Should save and read raw bytes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = LocalFolderContext(path=tmpdir, kwargs={})

            data = b"Hello, World!"
            ctx.save_file("test.txt", data)

            assert ctx.exists("test.txt")
            loaded = ctx.read_file("test.txt")
            assert loaded == data

    def test_save_and_read_xarray(self):
        """Should save and read xarray Datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = LocalFolderContext(path=tmpdir, kwargs={})

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
            ctx = LocalFolderContext(path=tmpdir, kwargs={})

            ctx.save_json("test.json", {"key": "value"})

            with ctx.open_file("test.json", "r") as f:
                content = f.read()
                assert "key" in content
                assert "value" in content

    def test_exists_false_for_missing(self):
        """exists() should return False for missing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = LocalFolderContext(path=tmpdir, kwargs={})
            assert not ctx.exists("nonexistent.json")

    def test_nested_directories(self):
        """Should handle nested directory paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = LocalFolderContext(path=tmpdir, kwargs={})

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
            dep_ctx = LocalFolderContext(path=str(dep_path), kwargs={})
            dep_ctx.save_json("result.json", {"dep_value": 123})

            # Create main context with dependency
            main_path = Path(tmpdir) / "main"
            main_ctx = LocalFolderContext(
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
            ctx = LocalFolderContext(path=tmpdir, kwargs={})

            with pytest.raises(KeyError):
                ctx.dependency("unknown")

    def test_creates_directory_if_missing(self):
        """Should create the directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "new_dir"
            assert not path.exists()

            ctx = LocalFolderContext(path=str(path), kwargs={})
            assert path.exists()


class TestOutputGuards:
    """Tests for output file validation."""

    def test_allowed_outputs_none_allows_all(self):
        """With allowed_outputs=None, all writes should be allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = LocalFolderContext(path=tmpdir, kwargs={}, allowed_outputs=None)
            assert ctx.allowed_outputs is None

            # All writes should work
            ctx.save_json("any_file.json", {"key": "value"})
            ctx.save_file("any_file.bin", b"data")
            assert ctx.exists("any_file.json")
            assert ctx.exists("any_file.bin")

    def test_allowed_outputs_restricts_writes(self):
        """With allowed_outputs set, only declared files can be written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = LocalFolderContext(
                path=tmpdir,
                kwargs={},
                allowed_outputs={"result.json", "model.nc"},
            )
            assert ctx.allowed_outputs == {"result.json", "model.nc"}

            # Allowed writes should work
            ctx.save_json("result.json", {"key": "value"})
            assert ctx.exists("result.json")

            # Disallowed writes should raise PermissionError
            with pytest.raises(PermissionError, match="undeclared.json"):
                ctx.save_json("undeclared.json", {"key": "value"})

    def test_allowed_outputs_empty_set_is_read_only(self):
        """With allowed_outputs=set(), context should be read-only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a file first
            ctx_write = LocalFolderContext(path=tmpdir, kwargs={})
            ctx_write.save_json("existing.json", {"key": "value"})

            # Create read-only context
            ctx_readonly = LocalFolderContext(
                path=tmpdir,
                kwargs={},
                allowed_outputs=set(),
            )

            # Reading should work
            data = ctx_readonly.read_json("existing.json")
            assert data == {"key": "value"}

            # Writing should raise PermissionError
            with pytest.raises(PermissionError, match="read-only"):
                ctx_readonly.save_json("new.json", {"key": "value"})

    def test_dependency_context_is_read_only(self):
        """Dependency contexts should be read-only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dependency output
            dep_path = Path(tmpdir) / "dependency"
            dep_path.mkdir()
            dep_ctx = LocalFolderContext(path=str(dep_path), kwargs={})
            dep_ctx.save_json("result.json", {"dep_value": 123})

            # Create main context with dependency
            main_path = Path(tmpdir) / "main"
            main_ctx = LocalFolderContext(
                path=str(main_path),
                kwargs={},
                dependency_paths={"dep1": str(dep_path)},
            )

            # Get dependency context
            dep = main_ctx.dependency("dep1")

            # Reading should work
            result = dep.read_json("result.json")
            assert result == {"dep_value": 123}

            # Writing should fail - dependency context is read-only
            with pytest.raises(PermissionError, match="read-only"):
                dep.save_json("new.json", {"key": "value"})

    def test_output_validation_on_save_file(self):
        """save_file should validate against allowed_outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = LocalFolderContext(
                path=tmpdir,
                kwargs={},
                allowed_outputs={"allowed.bin"},
            )

            ctx.save_file("allowed.bin", b"data")
            assert ctx.exists("allowed.bin")

            with pytest.raises(PermissionError):
                ctx.save_file("notallowed.bin", b"data")

    def test_output_validation_on_save_xarray(self):
        """save_xarray should validate against allowed_outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = LocalFolderContext(
                path=tmpdir,
                kwargs={},
                allowed_outputs={"allowed.nc"},
            )

            ds = xr.Dataset({"data": (["x"], [1, 2, 3])})
            ctx.save_xarray("allowed.nc", ds)
            assert ctx.exists("allowed.nc")

            with pytest.raises(PermissionError):
                ctx.save_xarray("notallowed.nc", ds)
