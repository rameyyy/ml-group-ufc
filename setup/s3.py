import os
from pathlib import Path
import boto3
from botocore.exceptions import ClientError


def _join_key(*parts: str) -> str:
    valid_parts = [str(p).strip("/") for p in parts if p]
    return "/".join(valid_parts)


def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )


def download_file(s3_key: str, local_path: str) -> None:
    bucket = os.getenv("S3_BUCKET_NAME")
    prefix = os.getenv("S3_PREFIX", "")

    if not bucket:
        raise ValueError("S3_BUCKET_NAME not set in environment")

    full_key = _join_key(prefix, s3_key)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    s3 = get_s3_client()
    try:
        s3.download_file(bucket, full_key, local_path)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "404":
            raise FileNotFoundError(f"S3 object not found: s3://{bucket}/{full_key}")
        raise


def list_objects(prefix: str = "") -> list:
    bucket = os.getenv("S3_BUCKET_NAME")
    base_prefix = os.getenv("S3_PREFIX", "")

    if not bucket:
        raise ValueError("S3_BUCKET_NAME not set in environment")

    full_prefix = _join_key(base_prefix, prefix)
    s3 = get_s3_client()

    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=full_prefix)
        if "Contents" not in response:
            return []

        keys = []
        for obj in response["Contents"]:
            key = obj["Key"]
            if base_prefix and key.startswith(base_prefix):
                key = key[len(base_prefix):].lstrip("/")
            keys.append(key)
        return keys

    except ClientError as e:
        print(f"Error listing objects: {e}")
        return []


def upload_file(local_path: str, s3_key: str) -> None:
    bucket = os.getenv("S3_BUCKET_NAME")
    prefix = os.getenv("S3_PREFIX", "")

    if not bucket:
        raise ValueError("S3_BUCKET_NAME not set in environment")

    if not Path(local_path).exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    full_key = _join_key(prefix, s3_key)
    s3 = get_s3_client()
    s3.upload_file(local_path, bucket, full_key)


def delete_file(s3_key: str) -> None:
    bucket = os.getenv("S3_BUCKET_NAME")
    prefix = os.getenv("S3_PREFIX", "")

    if not bucket:
        raise ValueError("S3_BUCKET_NAME not set in environment")

    full_key = _join_key(prefix, s3_key)
    s3 = get_s3_client()
    s3.delete_object(Bucket=bucket, Key=full_key)
