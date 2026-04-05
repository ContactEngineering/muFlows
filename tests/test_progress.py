"""Tests for muflow.storage.progress — ProgressChecker implementations."""

import json
import tempfile
from pathlib import Path

import pytest

from muflow.storage.progress import (
    LocalProgressChecker,
    ProgressChecker,
    make_progress_checker,
)

try:
    import boto3
    from moto import mock_aws
    HAS_S3_DEPS = True
except ImportError:
    HAS_S3_DEPS = False

BUCKET = "test-bucket"


# ── LocalProgressChecker ──────────────────────────────────────────────────────


class TestLocalProgressChecker:
    def test_empty_prefixes_returns_empty_set(self, tmp_path):
        checker = LocalProgressChecker()
        assert checker.completed_prefixes([]) == set()

    def test_prefix_without_manifest_not_returned(self, tmp_path):
        d = tmp_path / "node"
        d.mkdir()
        checker = LocalProgressChecker()
        assert checker.completed_prefixes([str(d)]) == set()

    def test_prefix_with_manifest_returned(self, tmp_path):
        d = tmp_path / "node"
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({"files": [], "timestamp": "t"}))
        checker = LocalProgressChecker()
        assert checker.completed_prefixes([str(d)]) == {str(d)}

    def test_mixed_prefixes(self, tmp_path):
        complete = tmp_path / "complete"
        incomplete = tmp_path / "incomplete"
        complete.mkdir()
        incomplete.mkdir()
        (complete / "manifest.json").write_text("{}")

        checker = LocalProgressChecker()
        result = checker.completed_prefixes([str(complete), str(incomplete)])
        assert result == {str(complete)}

    def test_nonexistent_prefix_not_returned(self, tmp_path):
        checker = LocalProgressChecker()
        assert checker.completed_prefixes([str(tmp_path / "ghost")]) == set()

    def test_to_config_returns_empty_dict(self):
        assert LocalProgressChecker().to_config() == {}

    def test_from_config_roundtrip(self):
        checker = LocalProgressChecker()
        restored = LocalProgressChecker.from_config(checker.to_config())
        assert isinstance(restored, LocalProgressChecker)

    def test_implements_protocol(self):
        assert isinstance(LocalProgressChecker(), ProgressChecker)


# ── make_progress_checker ─────────────────────────────────────────────────────


class TestMakeProgressChecker:
    def test_local_returns_local_checker(self):
        checker = make_progress_checker("local", {})
        assert isinstance(checker, LocalProgressChecker)

    def test_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown storage_type"):
            make_progress_checker("redis", {})

    @pytest.mark.skipif(not HAS_S3_DEPS, reason="boto3/moto required")
    def test_s3_returns_s3_checker(self):
        from muflow.storage.progress import S3ProgressChecker
        checker = make_progress_checker("s3", {"bucket": "b"})
        assert isinstance(checker, S3ProgressChecker)


# ── S3ProgressChecker ─────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_S3_DEPS, reason="boto3 and moto required")
class TestS3ProgressChecker:
    from muflow.storage.progress import S3ProgressChecker

    @pytest.fixture
    def s3_env(self):
        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket=BUCKET)
            yield s3

    def _write_manifest(self, s3, prefix: str) -> None:
        s3.put_object(
            Bucket=BUCKET,
            Key=f"{prefix}/manifest.json",
            Body=json.dumps({"files": [], "timestamp": "t"}).encode(),
        )

    def test_empty_prefixes_returns_empty_set(self, s3_env):
        from muflow.storage.progress import S3ProgressChecker
        checker = S3ProgressChecker(BUCKET)
        assert checker.completed_prefixes([]) == set()

    def test_prefix_without_manifest_not_returned(self, s3_env):
        from muflow.storage.progress import S3ProgressChecker
        checker = S3ProgressChecker(BUCKET)
        assert checker.completed_prefixes(["muflow/task/aaa"]) == set()

    def test_prefix_with_manifest_returned(self, s3_env):
        from muflow.storage.progress import S3ProgressChecker
        self._write_manifest(s3_env, "muflow/task/aaa")
        checker = S3ProgressChecker(BUCKET)
        assert checker.completed_prefixes(["muflow/task/aaa"]) == {"muflow/task/aaa"}

    def test_mixed_prefixes(self, s3_env):
        from muflow.storage.progress import S3ProgressChecker
        self._write_manifest(s3_env, "muflow/task/done")
        checker = S3ProgressChecker(BUCKET)
        result = checker.completed_prefixes(["muflow/task/done", "muflow/task/pending"])
        assert result == {"muflow/task/done"}

    def test_to_config_contains_bucket(self):
        from muflow.storage.progress import S3ProgressChecker
        assert S3ProgressChecker("my-bucket").to_config() == {"bucket": "my-bucket"}

    def test_from_config_roundtrip(self):
        from muflow.storage.progress import S3ProgressChecker
        checker = S3ProgressChecker("my-bucket")
        restored = S3ProgressChecker.from_config(checker.to_config())
        assert restored._bucket == "my-bucket"

    def test_implements_protocol(self):
        from muflow.storage.progress import S3ProgressChecker
        assert isinstance(S3ProgressChecker("b"), ProgressChecker)

    def test_non_404_error_propagates(self, s3_env):
        """Errors other than 404/NoSuchKey should re-raise."""
        from unittest.mock import patch

        from botocore.exceptions import ClientError

        from muflow.storage.progress import S3ProgressChecker

        error_response = {"Error": {"Code": "AccessDenied", "Message": "Denied"}}

        with patch.object(
            s3_env.__class__, "head_object",
            side_effect=ClientError(error_response, "HeadObject"),
        ):
            checker = S3ProgressChecker(BUCKET)
            # Patch at boto3 client level
            import boto3 as _boto3
            with patch.object(_boto3, "client", return_value=s3_env):
                with pytest.raises(ClientError, match="AccessDenied"):
                    checker.completed_prefixes(["muflow/task/aaa"])
