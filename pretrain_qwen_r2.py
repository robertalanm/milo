#!/usr/bin/env python
"""
Pretrain Qwen-2.5 1.5B with R2 streaming support.

This script extends LitGPT's pretraining functionality to support
streaming data from Cloudflare R2 (S3-compatible storage) in both
LitData format and Parquet format.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional
import yaml
from litgpt import Config
from litgpt.pretrain import setup
from litdata import StreamingDataset, StreamingDataLoader, TokensLoader
from transformers import AutoTokenizer
import torch

from r2_parquet_dataset import R2ParquetDataModule


class R2StreamingDataModule:
    """Custom data module for streaming from R2 bucket (LitData format)."""
    
    def __init__(
        self,
        data_path: str,
        block_size: int = 2048,
        batch_size: int = 64,
        num_workers: int = 8,
        storage_options: Optional[Dict[str, str]] = None,
    ):
        self.data_path = data_path
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.storage_options = storage_options or {}
        
    def setup(self):
        """Set up the streaming datasets."""
        # Add 1 to block_size for next-token prediction
        self.train_dataset = StreamingDataset(
            input_dir=f"{self.data_path}/train",
            item_loader=TokensLoader(block_size=self.block_size + 1),
            shuffle=True,
            drop_last=True,
            storage_options=self.storage_options,
        )
        
        # Validation dataset (optional)
        val_path = f"{self.data_path}/val"
        try:
            self.val_dataset = StreamingDataset(
                input_dir=val_path,
                item_loader=TokensLoader(block_size=self.block_size + 1),
                shuffle=False,
                drop_last=False,
                storage_options=self.storage_options,
            )
        except:
            print(f"No validation dataset found at {val_path}")
            self.val_dataset = None
    
    def train_dataloader(self):
        """Create training dataloader."""
        return StreamingDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
    
    def val_dataloader(self):
        """Create validation dataloader."""
        if self.val_dataset is None:
            return None
        return StreamingDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
        )


def main():
    """Main pretraining function with R2 support."""
    
    # Load configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/pretrain_qwen2.5_1.5b.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up R2 storage options
    # These should be set as environment variables for security
    storage_options = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL", "https://YOUR-ACCOUNT-ID.r2.cloudflarestorage.com"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY"),
    }
    
    # Validate R2 credentials
    if not storage_options["aws_access_key_id"] or not storage_options["aws_secret_access_key"]:
        raise ValueError(
            "R2 credentials not found. Please set R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY environment variables."
        )
    
    # Extract data configuration
    data_config = config.get("data", {})
    
    # Check if using Parquet format
    if isinstance(data_config, dict) and data_config.get("format") == "parquet":
        print("Using Parquet data format")
        
        # Extract bucket and path from data.path
        data_path = data_config.get("path", "")
        if not data_path.startswith("s3://"):
            raise ValueError(f"Data path must be an S3 URI, got: {data_path}")
        
        # Parse bucket and path
        path_parts = data_path.replace("s3://", "").split("/", 1)
        bucket_name = path_parts[0]
        dataset_path = path_parts[1] if len(path_parts) > 1 else ""
        
        # Load tokenizer
        tokenizer_name = data_config.get("tokenizer", "Qwen/Qwen2.5-1.5B")
        print(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        
        # Create Parquet data module
        data_module = R2ParquetDataModule(
            bucket_name=bucket_name,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            batch_size=config["train"]["micro_batch_size"],
            sequence_length=config["train"]["max_seq_length"],
            num_workers=data_config.get("num_workers", min(8, os.cpu_count())),
            pack_samples=data_config.get("pack_samples", True),
            r2_credentials={
                "endpoint_url": storage_options["endpoint_url"],
                "access_key_id": storage_options["aws_access_key_id"],
                "secret_access_key": storage_options["aws_secret_access_key"],
            }
        )
    else:
        print("Using LitData format")
        
        # Extract data path from config
        data_path = config.get("data.path", "").replace("data.path", "").strip()
        if not data_path.startswith("s3://"):
            raise ValueError(f"Data path must be an S3 URI, got: {data_path}")
        
        # Create custom data module for LitData format
        data_module = R2StreamingDataModule(
            data_path=data_path,
            block_size=config["train"]["max_seq_length"],
            batch_size=config["train"]["micro_batch_size"],
            num_workers=min(8, os.cpu_count()),
            storage_options=storage_options,
        )
    
    # Override the data module in config
    config["data"] = data_module
    
    # Run LitGPT pretraining
    setup(**config)


if __name__ == "__main__":
    main() 