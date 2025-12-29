#!/usr/bin/env python3
"""Quick S3 connectivity check for Yandex Object Storage."""

import os
import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from cassava_leaf_disease.training.train import _ensure_s3_env_from_dotenv  # noqa: E402

_ensure_s3_env_from_dotenv(repo_root=repo_root)

access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("YC_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("YC_SECRET_ACCESS_KEY")

if not access_key or not secret_key:
    print("[ERROR] S3 credentials not found in environment")
    print("   Check .env file or export AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
    sys.exit(1)

try:
    import boto3
except ImportError:
    print("[ERROR] boto3 not installed")
    sys.exit(1)

endpoint_url = "https://storage.yandexcloud.net"
bucket = "mlops-cassava-project"
region = "ru-central1"

print("Checking S3 connectivity...")
print(f"   Endpoint: {endpoint_url}")
print(f"   Bucket: {bucket}")
print(f"   Region: {region}")

try:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url,
        region_name=region,
    )

    # Test 1: List buckets
    print("\nTest 1: List buckets...")
    response = s3.list_buckets()
    buckets = [b["Name"] for b in response.get("Buckets", [])]
    print(f"   [OK] Found {len(buckets)} bucket(s): {', '.join(buckets)}")

    # Test 2: Check bucket access
    print(f"\nTest 2: Check bucket '{bucket}' access...")
    s3.head_bucket(Bucket=bucket)
    print(f"   [OK] Bucket '{bucket}' is accessible")

    # Test 3: List objects (first 5)
    print(f"\nTest 3: List objects in '{bucket}'...")
    response = s3.list_objects_v2(Bucket=bucket, MaxKeys=5)
    objects = response.get("Contents", [])
    if objects:
        print(f"   [OK] Found {len(objects)} object(s) (showing first 5):")
        for obj in objects:
            size_mb = obj["Size"] / (1024 * 1024)
            print(f"      - {obj['Key']} ({size_mb:.2f} MB)")
    else:
        print("   [WARN] Bucket is empty or no access to list objects")

    print("\n[SUCCESS] S3 connectivity check passed!")
    sys.exit(0)

except Exception as exc:
    print(f"\n[ERROR] S3 connectivity check failed: {exc}")
    sys.exit(1)
