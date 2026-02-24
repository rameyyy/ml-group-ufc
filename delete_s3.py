#!/usr/bin/env python3
"""Delete a file or folder from S3 bucket."""

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
    from setup.s3 import delete_file, list_objects
except ImportError:
    print("Error: Cannot import S3 utilities")
    sys.exit(1)


def delete_path(path: str):
    """Delete a file or all files in a folder from S3."""
    print("=" * 80)
    print("DELETING FROM S3")
    print("=" * 80)

    print(f"\nLooking for: {path}")

    all_files = list_objects()
    matching_files = [f for f in all_files if f == path or f.startswith(f"{path}/")]

    if not matching_files:
        print(f"No files found matching: {path}")
        return

    print(f"Found {len(matching_files)} file(s) to delete:\n")
    for f in matching_files:
        print(f"  - {f}")

    confirm = input("\nAre you sure you want to delete these files? (yes/no): ")
    if confirm.lower() != "yes":
        print("Cancelled.")
        return

    print("\nDeleting...")
    success_count = 0
    error_count = 0

    for s3_key in matching_files:
        try:
            delete_file(s3_key)
            print(f"Deleted: {s3_key}")
            success_count += 1
        except Exception as e:
            print(f"Error deleting {s3_key}: {e}")
            error_count += 1

    print("\n" + "=" * 80)
    print(f"Deleted: {success_count} | Errors: {error_count}")
    print("=" * 80)


if __name__ == "__main__":
    # Specify the file or folder to delete (relative to data/)
    # Examples:
    #   "sample_fight.json"          - deletes single file
    #   "models/model1.pkl"          - deletes single file in subfolder
    #   "models"                     - deletes entire models/ folder

    path_to_delete = "sample_fight.json"  # CHANGE THIS

    delete_path(path_to_delete)
