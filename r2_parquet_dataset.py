#!/usr/bin/env python
"""
R2 Parquet Dataset Loader for LitGPT pretraining.

This module provides efficient streaming of Parquet files from Cloudflare R2 storage,
based on the original R2DatasetLoader but simplified and adapted for LitGPT.
"""

import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
import logging

import numpy as np
import pyarrow
import pyarrow.parquet as pq
import s3fs
import yaml
import torch
from torch.utils.data import IterableDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set PyArrow thread count
pyarrow.set_io_thread_count(os.cpu_count())


class R2ParquetDataset(IterableDataset):
    """
    A PyTorch IterableDataset that reads Parquet files from Cloudflare R2 storage.
    
    This dataset handles:
    - Reading and caching metadata from R2 storage
    - Loading data from Parquet files in parallel
    - Tokenizing and batching text data
    - Managing sequence padding and packing
    """
    
    # Class-level configuration
    CF_REGION_NAME = "enam"
    MAX_CONCURRENT_REQUESTS = 32
    READ_BUFFER_SIZE = 32 * 1024 * 1024  # 32MB
    
    # Class-level caches
    _metadata_cache = {}
    _parquet_cache = {}
    _token_cache = {}
    _fs_cache = {}
    _fs_lock = threading.Lock()
    _executor = None
    
    def __init__(
        self,
        bucket_name: str,
        dataset_path: str,
        tokenizer,
        sequence_length: int = 2048,
        pack_samples: bool = True,
        num_workers: int = 8,
        r2_credentials: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the R2 Parquet dataset.
        
        Args:
            bucket_name: R2 bucket name
            dataset_path: Path within bucket to dataset
            tokenizer: Tokenizer instance
            sequence_length: Length of sequences to generate
            pack_samples: Whether to pack samples without padding
            num_workers: Number of parallel workers
            r2_credentials: Dict with endpoint_url, access_key_id, secret_access_key
        """
        self.bucket_name = bucket_name
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.pack_samples = pack_samples
        self.num_workers = num_workers
        
        # R2 credentials
        if r2_credentials:
            self.r2_credentials = r2_credentials
        else:
            self.r2_credentials = {
                "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
                "access_key_id": os.getenv("R2_ACCESS_KEY_ID"),
                "secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY"),
            }
        
        # Validate credentials
        if not all(self.r2_credentials.values()):
            raise ValueError("R2 credentials not complete. Need endpoint_url, access_key_id, and secret_access_key")
        
        # Initialize metadata
        self._metadata = None
        self._shard_sizes = None
        self._total_shards = 0
        
        # Buffers for tokenization
        self.buffer = []
        self.padded_buffer = []
        
        # Load metadata
        self._load_metadata()
    
    @classmethod
    def get_executor(cls):
        """Get or create a shared ThreadPoolExecutor."""
        if cls._executor is None:
            cls._executor = ThreadPoolExecutor(
                max_workers=cls.MAX_CONCURRENT_REQUESTS,
                thread_name_prefix="R2ParquetDataset",
            )
        return cls._executor
    
    def _get_fs(self):
        """Get or create S3FileSystem for R2."""
        with self._fs_lock:
            cache_key = self.r2_credentials["endpoint_url"]
            
            if cache_key not in self._fs_cache:
                fs = s3fs.S3FileSystem(
                    key=self.r2_credentials["access_key_id"],
                    secret=self.r2_credentials["secret_access_key"],
                    client_kwargs={
                        "endpoint_url": self.r2_credentials["endpoint_url"],
                        "region_name": self.CF_REGION_NAME,
                    },
                    config_kwargs={
                        "tcp_keepalive": True,
                        "max_pool_connections": 50,
                        "connect_timeout": 5,
                        "read_timeout": 10,
                        "retries": {"max_attempts": 3},
                    },
                    max_concurrency=self.MAX_CONCURRENT_REQUESTS,
                    use_listings_cache=True,
                    default_block_size=self.READ_BUFFER_SIZE,
                )
                self._fs_cache[cache_key] = fs
            
            return self._fs_cache[cache_key]
    
    def _load_metadata(self):
        """Load metadata from R2 storage."""
        fs = self._get_fs()
        
        # Cache directory for metadata
        cache_dir = Path(".cache/r2_dataset")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths
        bucket_path = f"{self.bucket_name}/{self.dataset_path}"
        metadata_files = {
            "shard_sizes": f"{bucket_path}/_shard_sizes.json",
            "metadata": f"{bucket_path}/_metadata.yaml",
        }
        
        try:
            # Load shard sizes
            local_shard_sizes = cache_dir / "shard_sizes.json"
            if not local_shard_sizes.exists():
                logger.info("Downloading shard sizes from R2...")
                fs.get(metadata_files["shard_sizes"], str(local_shard_sizes))
            
            with open(local_shard_sizes) as f:
                self._shard_sizes = json.load(f)
            
            # Load metadata config
            local_metadata = cache_dir / "metadata.yaml"
            if not local_metadata.exists():
                logger.info("Downloading metadata from R2...")
                fs.get(metadata_files["metadata"], str(local_metadata))
            
            with open(local_metadata) as f:
                self._metadata = yaml.safe_load(f)
            
            # Count total shards
            self._total_shards = sum(
                len(config_data.get("shards", [])) 
                for config_data in self._shard_sizes.values()
            )
            
            logger.info(f"Loaded metadata: {self._total_shards} total shards")
            
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise
    
    def _get_shard_paths(self) -> List[str]:
        """Get all shard paths from metadata."""
        paths = []
        for config_name, config_data in self._shard_sizes.items():
            for shard in config_data.get("shards", []):
                if "path" in shard:
                    paths.append(shard["path"])
        return paths
    
    @lru_cache(maxsize=128)
    def _get_parquet_file(self, shard_path: str) -> dict:
        """Get cached ParquetFile object."""
        fs = self._get_fs()
        
        try:
            # Open file
            f = fs.open(shard_path, "rb", buffer_size=self.READ_BUFFER_SIZE)
            pf = pq.ParquetFile(
                f,
                memory_map=False,
                pre_buffer=True,
                buffer_size=self.READ_BUFFER_SIZE,
            )
            
            return {
                "file": f,
                "parquet": pf,
                "lock": threading.Lock(),
                "metadata": {
                    "path": shard_path,
                    "num_row_groups": pf.num_row_groups,
                    "total_rows": pf.metadata.num_rows,
                },
            }
        except Exception as e:
            logger.error(f"Failed to open parquet file {shard_path}: {e}")
            raise
    
    def _read_row_group(self, pf_data: dict, group_index: int) -> pyarrow.Table:
        """Read a specific row group from parquet file."""
        with pf_data["lock"]:
            if pf_data["file"].closed:
                raise IOError(f"Parquet file is closed")
            
            return pf_data["parquet"].read_row_group(
                group_index,
                columns=["text"],
                use_threads=False,
            )
    
    def _tokenize_texts(self, texts: List[str]) -> List[int]:
        """Tokenize a batch of texts."""
        all_tokens = []
        
        # Batch tokenization
        batch_size = 128
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize batch
            encoded = self.tokenizer(
                batch,
                padding=False,
                truncation=False,
                return_tensors=None,
            )
            
            # Extract tokens and add EOS if needed
            for tokens in encoded["input_ids"]:
                if tokens:
                    all_tokens.extend(tokens)
                    if tokens[-1] != self.tokenizer.eos_token_id:
                        all_tokens.append(self.tokenizer.eos_token_id)
        
        return all_tokens
    
    def _process_shard(self, shard_path: str) -> Iterator[List[int]]:
        """Process a single shard and yield tokenized data."""
        try:
            # Get parquet file
            pf_data = self._get_parquet_file(shard_path)
            
            # Process each row group
            for group_idx in range(pf_data["metadata"]["num_row_groups"]):
                # Read row group
                table = self._read_row_group(pf_data, group_idx)
                
                # Extract texts
                texts = table["text"].to_pylist()
                
                # Tokenize
                tokens = self._tokenize_texts(texts)
                
                if tokens:
                    yield tokens
                    
        except Exception as e:
            logger.error(f"Error processing shard {shard_path}: {e}")
            raise
    
    def __iter__(self):
        """Iterate over the dataset."""
        # Get worker info for sharding
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single process
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        
        # Get shard paths and distribute among workers
        shard_paths = self._get_shard_paths()
        worker_shards = [
            path for i, path in enumerate(shard_paths) 
            if i % num_workers == worker_id
        ]
        
        logger.info(f"Worker {worker_id}: Processing {len(worker_shards)} shards")
        
        # Reset buffers
        self.buffer = []
        self.padded_buffer = []
        
        # Process shards
        for shard_path in worker_shards:
            for tokens in self._process_shard(shard_path):
                self.buffer.extend(tokens)
                
                # Yield sequences when buffer is large enough
                while len(self.buffer) >= self.sequence_length:
                    if self.pack_samples:
                        # Pack without padding
                        yield self.buffer[:self.sequence_length]
                        self.buffer = self.buffer[self.sequence_length:]
                    else:
                        # Find next EOS token for proper sequence boundary
                        try:
                            eos_idx = self.buffer.index(self.tokenizer.eos_token_id)
                            sequence = self.buffer[:eos_idx + 1]
                            self.buffer = self.buffer[eos_idx + 1:]
                            
                            # Pad sequence
                            if len(sequence) < self.sequence_length:
                                padding = [self.tokenizer.pad_token_id] * (self.sequence_length - len(sequence))
                                sequence.extend(padding)
                            elif len(sequence) > self.sequence_length:
                                sequence = sequence[:self.sequence_length]
                            
                            yield sequence
                        except ValueError:
                            # No EOS found, take full sequence
                            yield self.buffer[:self.sequence_length]
                            self.buffer = self.buffer[self.sequence_length:]


class R2ParquetDataModule:
    """Data module for R2 Parquet datasets compatible with LitGPT."""
    
    def __init__(
        self,
        bucket_name: str,
        dataset_path: str,
        tokenizer,
        batch_size: int = 64,
        sequence_length: int = 2048,
        num_workers: int = 8,
        pack_samples: bool = True,
        r2_credentials: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the data module.
        
        Args:
            bucket_name: R2 bucket name
            dataset_path: Path within bucket to dataset
            tokenizer: Tokenizer instance
            batch_size: Batch size for training
            sequence_length: Sequence length
            num_workers: Number of data loading workers
            pack_samples: Whether to pack samples
            r2_credentials: R2 credentials dict
        """
        self.bucket_name = bucket_name
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_workers = num_workers
        self.pack_samples = pack_samples
        self.r2_credentials = r2_credentials
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage: str = "fit"):
        """Set up datasets."""
        if stage == "fit":
            # Training dataset
            self.train_dataset = R2ParquetDataset(
                bucket_name=self.bucket_name,
                dataset_path=self.dataset_path,
                tokenizer=self.tokenizer,
                sequence_length=self.sequence_length,
                pack_samples=self.pack_samples,
                num_workers=self.num_workers,
                r2_credentials=self.r2_credentials,
            )
            
            # Note: This implementation assumes all data is for training
            # For validation, you would need separate paths or logic
    
    def train_dataloader(self):
        """Get training dataloader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def val_dataloader(self):
        """Get validation dataloader."""
        # Return None for now - would need separate validation data
        return None 