#!/usr/bin/env python
"""
Test script for R2 Parquet dataset functionality.
"""

import os
import sys
import time
from transformers import AutoTokenizer
import torch
from r2_parquet_dataset import R2ParquetDataset, R2ParquetDataModule


def test_parquet_dataset(bucket_name: str, dataset_path: str):
    """Test the Parquet dataset loading."""
    print(f"Testing Parquet dataset from s3://{bucket_name}/{dataset_path}")
    
    # R2 credentials
    r2_credentials = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "access_key_id": os.getenv("R2_ACCESS_KEY_ID"),
        "secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY"),
    }
    
    # Validate credentials
    if not all(r2_credentials.values()):
        print("❌ R2 credentials not found!")
        print("Please set R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, and R2_SECRET_ACCESS_KEY")
        return False
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
        
        # Create dataset
        print("Creating Parquet dataset...")
        dataset = R2ParquetDataset(
            bucket_name=bucket_name,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            sequence_length=2048,
            pack_samples=True,
            num_workers=2,
            r2_credentials=r2_credentials,
        )
        
        print("✅ Successfully created Parquet dataset!")
        print(f"Total shards: {dataset._total_shards}")
        
        # Test iteration
        print("\nTesting data iteration...")
        start_time = time.time()
        
        for i, batch in enumerate(dataset):
            if i >= 5:  # Only test first 5 batches
                break
            
            print(f"Batch {i}: {len(batch)} tokens")
            print(f"  First 20 tokens: {batch[:20]}")
            
            # Verify sequence length
            if len(batch) != 2048:
                print(f"  ⚠️  Unexpected sequence length: {len(batch)}")
        
        elapsed = time.time() - start_time
        print(f"\n✅ Data iteration successful! ({elapsed:.2f}s for 5 batches)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Parquet dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_module(bucket_name: str, dataset_path: str):
    """Test the data module functionality."""
    print(f"\nTesting Parquet data module...")
    
    r2_credentials = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "access_key_id": os.getenv("R2_ACCESS_KEY_ID"),
        "secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY"),
    }
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
        
        # Create data module
        data_module = R2ParquetDataModule(
            bucket_name=bucket_name,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            batch_size=4,
            sequence_length=2048,
            num_workers=2,
            pack_samples=True,
            r2_credentials=r2_credentials,
        )
        
        # Setup datasets
        data_module.setup("fit")
        
        # Create dataloader
        train_loader = data_module.train_dataloader()
        
        print("✅ Data module created successfully!")
        
        # Test batched loading
        print("\nTesting batched data loading...")
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Only test first 3 batches
                break
            
            print(f"Batch {i}: shape {batch.shape}")
            
            # Verify batch dimensions
            expected_shape = (4, 2048)  # batch_size x sequence_length
            if batch.shape != expected_shape:
                print(f"  ⚠️  Unexpected batch shape: {batch.shape} (expected {expected_shape})")
        
        print("✅ Batched loading successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing data module: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("=== R2 Parquet Dataset Test ===\n")
    
    if len(sys.argv) < 2:
        print("Usage: python test_parquet_dataset.py s3://bucket/path/to/dataset")
        print("\nExample:")
        print("  python test_parquet_dataset.py s3://dclm-2/mlfoundations-dclm-baseline-1.0-parquet-optimized")
        sys.exit(1)
    
    # Parse S3 URI
    s3_uri = sys.argv[1]
    if not s3_uri.startswith("s3://"):
        print(f"Error: Invalid S3 URI format: {s3_uri}")
        print("Expected format: s3://bucket/path/to/dataset")
        sys.exit(1)
    
    # Extract bucket and path
    path_parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket_name = path_parts[0]
    dataset_path = path_parts[1] if len(path_parts) > 1 else ""
    
    print(f"Bucket: {bucket_name}")
    print(f"Dataset path: {dataset_path}")
    print()
    
    # Run tests
    success = True
    
    if not test_parquet_dataset(bucket_name, dataset_path):
        success = False
    
    if not test_data_module(bucket_name, dataset_path):
        success = False
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 