#!/usr/bin/env python3
"""Download all files from S3 bucket maintaining folder structure."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

env_path = Path("setup/.env")
if not env_path.exists():
    print("Error: setup/.env not found!")
    print("\nCreate setup/.env by copying setup/.env.example and fill in your credentials.")
    sys.exit(1)

load_dotenv(env_path)

required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_BUCKET_NAME"]
missing = [v for v in required_vars if not os.getenv(v)]
if missing:
    print(f"Error: Missing environment variables: {', '.join(missing)}")
    sys.exit(1)

try:
    from setup.s3 import download_file, list_objects
except ImportError:
    print("Error: Cannot import S3 utilities")
    sys.exit(1)


def main():
    print("=" * 80)
    print("DOWNLOADING ALL FILES FROM S3 BUCKET")
    print("=" * 80)

    bucket = os.getenv("S3_BUCKET_NAME")
    prefix = os.getenv("S3_PREFIX", "")

    print(f"\nBucket: {bucket}")
    print(f"Prefix: {prefix}")

    print("\nListing files...")
    files = list_objects()

    if not files:
        print("No files found in bucket.")
        return

    print(f"Found {len(files)} files\n")

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    success_count = 0
    skip_count = 0
    error_count = 0

    for s3_key in files:
        local_path = data_dir / s3_key

        if local_path.exists():
            print(f"SKIP: {s3_key} (already exists)")
            skip_count += 1
            continue

        try:
            print(f"Downloading: {s3_key}")
            download_file(s3_key, str(local_path))
            size_mb = local_path.stat().st_size / 1024 / 1024
            print(f"  Success ({size_mb:.1f} MB)")
            success_count += 1
        except Exception as e:
            print(f"  Error: {e}")
            error_count += 1

    print("\n" + "=" * 80)
    print(f"Downloaded: {success_count} | Skipped: {skip_count} | Errors: {error_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()
