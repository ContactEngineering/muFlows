"""Tests for S3 storage backend."""

import os
import pytest
import xarray as xr

from muflow.storage import S3StorageBackend, StorageBackend

try:
    import boto3
    from moto import mock_aws
    HAS_S3_DEPS = True
except ImportError:
    HAS_S3_DEPS = False


pytestmark = pytest.mark.skipif(
    not HAS_S3_DEPS,
    reason="boto3 and moto are required for S3 tests"
)


@pytest.fixture
def s3_bucket():
    endpoint_url = os.environ.get("AWS_S3_ENDPOINT_URL")
    if endpoint_url:
        # Use real S3 (e.g. Minio)
        bucket_name = os.environ.get("AWS_STORAGE_BUCKET_NAME", "test-bucket")
        s3 = boto3.resource(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "admin"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "secret12"),
            region_name="us-east-1",
        )
        # Ensure bucket exists
        bucket = s3.Bucket(bucket_name)
        try:
            bucket.create()
        except s3.meta.client.exceptions.BucketAlreadyOwnedByYou:
            pass
        yield bucket_name
    else:
        # Use moto
        with mock_aws():
            conn = boto3.resource("s3", region_name="us-east-1")
            bucket_name = "test-bucket"
            conn.create_bucket(Bucket=bucket_name)
            yield bucket_name


@pytest.fixture
def s3_client():
    endpoint_url = os.environ.get("AWS_S3_ENDPOINT_URL")
    if endpoint_url:
        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "admin"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "secret12"),
            region_name="us-east-1",
        )
    return None


class TestS3StorageBackend:
    def test_implements_protocol(self, s3_bucket, s3_client):
        backend = S3StorageBackend(
            storage_prefix="test", bucket=s3_bucket, s3_client=s3_client
        )
        assert isinstance(backend, StorageBackend)

    def test_save_read_json(self, s3_bucket, s3_client):
        backend = S3StorageBackend(
            storage_prefix="test", bucket=s3_bucket, s3_client=s3_client
        )
        data = {"key": "value", "n": 42}
        backend.save_json("data.json", data)
        assert backend.read_json("data.json") == data

    def test_save_read_file(self, s3_bucket, s3_client):
        backend = S3StorageBackend(
            storage_prefix="test", bucket=s3_bucket, s3_client=s3_client
        )
        data = b"hello world"
        backend.save_file("test.bin", data)
        assert backend.read_file("test.bin") == data

    def test_save_read_xarray(self, s3_bucket, s3_client):
        backend = S3StorageBackend(
            storage_prefix="test", bucket=s3_bucket, s3_client=s3_client
        )
        ds = xr.Dataset({"temp": (["x"], [1.0, 2.0, 3.0])})
        backend.save_xarray("model.nc", ds)
        result = backend.read_xarray("model.nc")
        xr.testing.assert_equal(ds, result)

    def test_exists(self, s3_bucket, s3_client):
        backend = S3StorageBackend(
            storage_prefix="test", bucket=s3_bucket, s3_client=s3_client
        )
        assert not backend.exists("nope.json")
        backend.save_json("data.json", {})
        assert backend.exists("data.json")

    def test_is_cached(self, s3_bucket, s3_client):
        backend = S3StorageBackend(
            storage_prefix="test", bucket=s3_bucket, s3_client=s3_client
        )
        assert not backend.is_cached()
        backend.write_manifest()
        assert backend.is_cached()

    def test_open_file(self, s3_bucket, s3_client):
        backend = S3StorageBackend(
            storage_prefix="test", bucket=s3_bucket, s3_client=s3_client
        )
        backend.save_file("text.txt", b"hello")
        with backend.open_file("text.txt", "r") as f:
            assert f.read() == "hello"

    def test_write_once_enforcement(self, s3_bucket, s3_client):
        backend = S3StorageBackend(
            storage_prefix="test", bucket=s3_bucket, s3_client=s3_client
        )
        backend.save_json("data.json", {"v": 1})
        with pytest.raises(FileExistsError):
            backend.save_json("data.json", {"v": 2})
