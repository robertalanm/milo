#!/usr/bin/env python
"""
Prepare and tokenize text data for streaming from R2.

This script:
1. Tokenizes raw text data using Qwen tokenizer
2. Optimizes it for streaming with LitData
3. Optionally uploads to R2 bucket
"""

import os
import json
from pathlib import Path
from functools import partial
from typing import List, Iterator
import boto3
from litdata import optimize, TokensLoader
from transformers import AutoTokenizer
import argparse


def tokenize_text_file(
    filepath: str,
    tokenizer,
    max_length: int = 2048,
    add_bos: bool = True,
    add_eos: bool = True
) -> Iterator[List[int]]:
    """
    Tokenize a text file and yield token chunks.
    
    Args:
        filepath: Path to the text file
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length
        add_bos: Add beginning of sequence token
        add_eos: Add end of sequence token
    
    Yields:
        List of token IDs
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text_buffer = ""
        
        for line in f:
            text_buffer += line
            
            # Tokenize when buffer is large enough
            if len(text_buffer) > max_length * 4:  # Rough estimate
                tokens = tokenizer.encode(text_buffer, add_special_tokens=False)
                
                # Add special tokens
                if add_bos and len(tokens) > 0:
                    tokens = [tokenizer.bos_token_id] + tokens
                if add_eos:
                    tokens = tokens + [tokenizer.eos_token_id]
                
                # Yield chunks of max_length
                for i in range(0, len(tokens) - max_length, max_length):
                    yield tokens[i:i + max_length + 1]  # +1 for next token prediction
                
                # Keep remainder for next iteration
                remainder_start = (len(tokens) // max_length) * max_length
                if remainder_start < len(tokens):
                    remainder_tokens = tokens[remainder_start:]
                    text_buffer = tokenizer.decode(remainder_tokens, skip_special_tokens=True)
                else:
                    text_buffer = ""
        
        # Process remaining text
        if text_buffer:
            tokens = tokenizer.encode(text_buffer, add_special_tokens=False)
            if add_bos and len(tokens) > 0:
                tokens = [tokenizer.bos_token_id] + tokens
            if add_eos:
                tokens = tokens + [tokenizer.eos_token_id]
            
            for i in range(0, len(tokens) - max_length, max_length):
                yield tokens[i:i + max_length + 1]


def tokenize_jsonl_file(
    filepath: str,
    tokenizer,
    text_key: str = "text",
    max_length: int = 2048,
    add_bos: bool = True,
    add_eos: bool = True
) -> Iterator[List[int]]:
    """
    Tokenize a JSONL file and yield token chunks.
    
    Args:
        filepath: Path to the JSONL file
        tokenizer: HuggingFace tokenizer instance
        text_key: Key in JSON containing text
        max_length: Maximum sequence length
        add_bos: Add beginning of sequence token
        add_eos: Add end of sequence token
    
    Yields:
        List of token IDs
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                text = data.get(text_key, "")
                
                if not text:
                    continue
                
                tokens = tokenizer.encode(text, add_special_tokens=False)
                
                # Add special tokens
                if add_bos:
                    tokens = [tokenizer.bos_token_id] + tokens
                if add_eos:
                    tokens = tokens + [tokenizer.eos_token_id]
                
                # Yield chunks of max_length
                for i in range(0, len(tokens) - max_length, max_length):
                    yield tokens[i:i + max_length + 1]
                    
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line in {filepath}")
                continue


def upload_to_r2(local_dir: str, r2_bucket: str, r2_prefix: str, storage_options: dict):
    """
    Upload optimized data to R2 bucket.
    
    Args:
        local_dir: Local directory containing optimized data
        r2_bucket: R2 bucket name
        r2_prefix: Prefix in R2 bucket
        storage_options: Storage options for R2
    """
    # Create S3 client with R2 endpoint
    s3_client = boto3.client(
        's3',
        endpoint_url=storage_options['endpoint_url'],
        aws_access_key_id=storage_options['aws_access_key_id'],
        aws_secret_access_key=storage_options['aws_secret_access_key']
    )
    
    # Upload all files in the directory
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = f"{r2_prefix}/{relative_path}"
            
            print(f"Uploading {local_path} to s3://{r2_bucket}/{s3_key}")
            s3_client.upload_file(local_path, r2_bucket, s3_key)


def main():
    parser = argparse.ArgumentParser(description="Prepare data for R2 streaming")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing raw text files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for optimized output")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-1.5B", help="HuggingFace tokenizer name")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--chunk-size", type=int, default=2049 * 8012, help="Chunk size for optimization")
    parser.add_argument("--file-type", type=str, choices=["txt", "jsonl"], default="txt", help="Input file type")
    parser.add_argument("--text-key", type=str, default="text", help="JSON key for text (for JSONL files)")
    parser.add_argument("--split", type=str, default="train", help="Data split name")
    
    # R2 upload options
    parser.add_argument("--upload-to-r2", action="store_true", help="Upload to R2 after optimization")
    parser.add_argument("--r2-bucket", type=str, help="R2 bucket name")
    parser.add_argument("--r2-prefix", type=str, help="R2 prefix/folder")
    
    args = parser.parse_args()
    
    # Initialize tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    
    # Collect input files
    input_files = []
    for ext in ["txt", "jsonl", "json"]:
        input_files.extend(Path(args.input_dir).rglob(f"*.{ext}"))
    
    if not input_files:
        raise ValueError(f"No text files found in {args.input_dir}")
    
    print(f"Found {len(input_files)} files to process")
    
    # Choose tokenization function based on file type
    if args.file_type == "jsonl":
        tokenize_fn = partial(
            tokenize_jsonl_file,
            tokenizer=tokenizer,
            text_key=args.text_key,
            max_length=args.max_length
        )
    else:
        tokenize_fn = partial(
            tokenize_text_file,
            tokenizer=tokenizer,
            max_length=args.max_length
        )
    
    # Optimize data
    output_dir = os.path.join(args.output_dir, args.split)
    print(f"Optimizing data to {output_dir}")
    
    optimize(
        fn=tokenize_fn,
        inputs=[str(f) for f in input_files],
        output_dir=output_dir,
        chunk_size=args.chunk_size,
        item_loader=TokensLoader(),
        num_workers=os.cpu_count(),
    )
    
    print("Data optimization complete!")
    
    # Upload to R2 if requested
    if args.upload_to_r2:
        if not args.r2_bucket or not args.r2_prefix:
            raise ValueError("--r2-bucket and --r2-prefix required for R2 upload")
        
        storage_options = {
            "endpoint_url": os.getenv("R2_ENDPOINT_URL", "https://YOUR-ACCOUNT-ID.r2.cloudflarestorage.com"),
            "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY"),
        }
        
        if not storage_options["aws_access_key_id"] or not storage_options["aws_secret_access_key"]:
            raise ValueError("R2 credentials not found. Set R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY")
        
        print(f"Uploading to R2 bucket: {args.r2_bucket}/{args.r2_prefix}")
        upload_to_r2(args.output_dir, args.r2_bucket, args.r2_prefix, storage_options)
        print("Upload complete!")


if __name__ == "__main__":
    main() 