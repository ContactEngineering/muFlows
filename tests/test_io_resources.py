"""Tests for muflow.io.resources module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from muflow.io.resources import ResourceManager, is_local_file, is_url, resolve_uri


class TestIsUrl:
    """Tests for is_url function."""

    def test_https_url(self):
        assert is_url("https://example.com/file.nc") is True

    def test_http_url(self):
        assert is_url("http://example.com/file.nc") is True

    def test_local_path(self):
        assert is_url("/path/to/file.nc") is False

    def test_relative_path(self):
        assert is_url("relative/path.nc") is False

    def test_file_uri(self):
        assert is_url("file:///path/to/file.nc") is False


class TestIsLocalFile:
    """Tests for is_local_file function."""

    def test_absolute_path(self):
        assert is_local_file("/path/to/file.nc") is True

    def test_relative_path(self):
        assert is_local_file("relative/path.nc") is True

    def test_file_uri(self):
        assert is_local_file("file:///path/to/file.nc") is True

    def test_https_url(self):
        assert is_local_file("https://example.com/file.nc") is False

    def test_http_url(self):
        assert is_local_file("http://example.com/file.nc") is False


class TestResolveUri:
    """Tests for resolve_uri function."""

    def test_local_path_unchanged(self):
        path = "/path/to/file.nc"
        assert resolve_uri(path) == path

    def test_relative_path_unchanged(self):
        path = "relative/file.nc"
        assert resolve_uri(path) == path

    def test_file_uri_extracts_path(self):
        uri = "file:///path/to/file.nc"
        assert resolve_uri(uri) == "/path/to/file.nc"

    def test_unsupported_scheme_raises(self):
        with pytest.raises(ValueError, match="Unsupported URI scheme"):
            resolve_uri("ftp://example.com/file.nc")

    @patch("muflow.io.resources.urllib.request.urlopen")
    def test_https_url_downloads(self, mock_urlopen):
        """Test that HTTPS URLs are downloaded to temp file."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"test content"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = resolve_uri("https://example.com/test.nc")

        assert Path(result).exists()
        assert Path(result).read_bytes() == b"test content"

        # Cleanup
        Path(result).unlink()


class TestResourceManager:
    """Tests for ResourceManager class."""

    def test_local_file_no_cleanup_needed(self):
        """Local files should not be cleaned up."""
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            f.write(b"test")
            local_path = f.name

        with ResourceManager() as rm:
            result = rm.resolve(local_path)
            assert result == local_path

        # File should still exist after context exit
        assert Path(local_path).exists()
        Path(local_path).unlink()

    @patch("muflow.io.resources.urllib.request.urlopen")
    def test_url_cleanup(self, mock_urlopen):
        """URL downloads should be cleaned up after context exit."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"test content"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with ResourceManager() as rm:
            result = rm.resolve("https://example.com/test.nc")
            temp_path = result
            assert Path(temp_path).exists()

        # File should be cleaned up after context exit
        assert not Path(temp_path).exists()

    @patch("muflow.io.resources.urllib.request.urlopen")
    def test_multiple_urls_all_cleaned_up(self, mock_urlopen):
        """Multiple URL downloads should all be cleaned up."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"test content"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        temp_paths = []
        with ResourceManager() as rm:
            for i in range(3):
                result = rm.resolve(f"https://example.com/test{i}.nc")
                temp_paths.append(result)
                assert Path(result).exists()

        # All should be cleaned up
        for path in temp_paths:
            assert not Path(path).exists()

    def test_file_uri_scheme(self):
        """file:// URIs should work."""
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            f.write(b"test")
            local_path = f.name

        with ResourceManager() as rm:
            result = rm.resolve(f"file://{local_path}")
            assert result == local_path

        Path(local_path).unlink()
