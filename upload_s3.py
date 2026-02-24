#!/usr/bin/env python3
"""Upload new files from data/ to S3 bucket maintaining folder structure."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

env_path = Path("setup/.env")
if not env_path.exists():
    print("Error: setup/.env not found!")
    sys.exit(1)

load_dotenv(env_path)

required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_BUCKET_NAME"]
missing = [v for v in required_vars if not os.getenv(v)]
if missing:
    print(f"Error: Missing environment variables: {', '.join(missing)}")
    sys.exit(1)

try:
    from setup.s3 import upload_file, list_objects
except ImportError:
    print("Error: Cannot import S3 utilities")
    sys.exit(1)


def main():
    print("=" * 80)
    print("UPLOADING NEW FILES TO S3")
    print("=" * 80)

    data_dir = Path("data")
    if not data_dir.exists():
        print("Error: data/ directory not found")
        return

    print("\nListing existing files in S3...")
    existing_files = set(list_objects())
    print(f"Found {len(existing_files)} existing files in bucket")

    print("\nScanning local data/ directory...")
    local_files = []
    for file_path in data_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix != ".md":
            relative_path = file_path.relative_to(data_dir)
            local_files.append((file_path, str(relative_path)))

    print(f"Found {len(local_files)} local files")

    new_files = [(fp, rp) for fp, rp in local_files if rp not in existing_files]

    if not new_files:
        print("\nNo new files to upload. All files already in S3.")
        return

    print(f"\nUploading {len(new_files)} new files...\n")

    success_count = 0
    error_count = 0

    for file_path, relative_path in new_files:
        try:
            print(f"Uploading: {relative_path}")
            upload_file(str(file_path), relative_path)
            size_mb = file_path.stat().st_size / 1024 / 1024
            print(f"  Success ({size_mb:.1f} MB)")
            success_count += 1
        except Exception as e:
            print(f"  Error: {e}")
            error_count += 1

    print("\n" + "=" * 80)
    print(f"Uploaded: {success_count} | Errors: {error_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()
