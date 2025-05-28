#!/usr/bin/env python
"""
Test script to verify R2 connection and data streaming.
"""

import os
import sys
from litdata import StreamingDataset, TokensLoader
import boto3
from botocore.exceptions import ClientError


def test_r2_connection():
    """Test R2 connection and credentials."""
    print("Testing R2 connection...")
    
    # Get credentials from environment
    storage_options = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY"),
    }
    
    # Validate credentials
    if not all(storage_options.values()):
        print("❌ R2 credentials not found!")
        print("Please set the following environment variables:")
        print("  export R2_ENDPOINT_URL=https://YOUR-ACCOUNT-ID.r2.cloudflarestorage.com")
        print("  export R2_ACCESS_KEY_ID=your_access_key")
        print("  export R2_SECRET_ACCESS_KEY=your_secret_key")
        return False
    
    # Test S3 client connection
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=storage_options['endpoint_url'],
            aws_access_key_id=storage_options['aws_access_key_id'],
            aws_secret_access_key=storage_options['aws_secret_access_key']
        )
        
        # Try to list buckets
        response = s3_client.list_buckets()
        print("✅ Successfully connected to R2!")
        print(f"Available buckets: {[b['Name'] for b in response['Buckets']]}")
        return True
        
    except ClientError as e:
        print(f"❌ Failed to connect to R2: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_data_streaming(data_path: str):
    """Test streaming data from R2."""
    print(f"\nTesting data streaming from: {data_path}")
    
    storage_options = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY"),
    }
    
    try:
        # Create streaming dataset
        dataset = StreamingDataset(
            input_dir=data_path,
            item_loader=TokensLoader(block_size=2049),
            shuffle=False,
            storage_options=storage_options,
        )
        
        print("✅ Successfully created StreamingDataset!")
        
        # Try to load a few samples
        print("\nLoading first 3 samples...")
        for i, sample in enumerate(dataset):
            if i >= 3:
                break
            print(f"Sample {i}: shape={sample.shape}, dtype={sample.dtype}")
            print(f"  First 10 tokens: {sample[:10].tolist()}")
            
        print("✅ Data streaming working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to stream data: {e}")
        return False


def main():
    """Main test function."""
    print("=== R2 Connection and Data Streaming Test ===\n")
    
    # Test R2 connection
    if not test_r2_connection():
        sys.exit(1)
    
    # Test data streaming if path provided
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        if not test_data_streaming(data_path):
            sys.exit(1)
    else:
        print("\nTo test data streaming, run:")
        print(f"  python {sys.argv[0]} s3://your-bucket/path/to/data/train")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main() 