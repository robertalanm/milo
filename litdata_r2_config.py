"""
LitData configuration for R2 streaming.

This module provides a data configuration that can be used with
the standard LitGPT CLI while supporting R2 streaming.
"""

import os
from typing import Optional, Dict
from litdata import StreamingDataset, StreamingDataLoader, TokensLoader


class LitDataR2:
    """LitData module configured for R2 streaming."""
    
    def __init__(
        self, 
        path: str,
        num_workers: int = 8,
        batch_size: Optional[int] = None,
        max_seq_length: int = 2048,
    ):
        """
        Initialize R2 streaming data module.
        
        Args:
            path: S3 URI to the data (e.g., s3://bucket/path)
            num_workers: Number of data loading workers
            batch_size: Batch size (if None, will be set by trainer)
            max_seq_length: Maximum sequence length
        """
        self.path = path
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        
        # R2 storage options from environment
        self.storage_options = {
            "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
            "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY"),
        }
        
        # Validate credentials
        if not all(self.storage_options.values()):
            raise ValueError(
                "R2 credentials not found. Please set R2_ENDPOINT_URL, "
                "R2_ACCESS_KEY_ID, and R2_SECRET_ACCESS_KEY environment variables."
            )
    
    def setup(self, stage: str = "fit"):
        """Set up datasets for training/validation."""
        if stage == "fit":
            # Training dataset
            self.train_dataset = StreamingDataset(
                input_dir=f"{self.path}/train",
                item_loader=TokensLoader(block_size=self.max_seq_length + 1),
                shuffle=True,
                drop_last=True,
                storage_options=self.storage_options,
            )
            
            # Validation dataset (optional)
            try:
                self.val_dataset = StreamingDataset(
                    input_dir=f"{self.path}/val",
                    item_loader=TokensLoader(block_size=self.max_seq_length + 1),
                    shuffle=False,
                    drop_last=False,
                    storage_options=self.storage_options,
                )
            except:
                self.val_dataset = None
    
    def train_dataloader(self) -> StreamingDataLoader:
        """Get training dataloader."""
        return StreamingDataLoader(
            self.train_dataset,
            batch_size=self.batch_size or 1,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
    
    def val_dataloader(self) -> Optional[StreamingDataLoader]:
        """Get validation dataloader."""
        if self.val_dataset is None:
            return None
        return StreamingDataLoader(
            self.val_dataset,
            batch_size=self.batch_size or 1,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
        )


# Register the data module so it can be used with LitGPT CLI
# This allows using --data LitDataR2 in the command line
__all__ = ["LitDataR2"] 