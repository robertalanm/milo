# Qwen-2.5 1.5B Pretraining with R2 Streaming

This project provides a complete setup for pretraining a Qwen-2.5 1.5B model using LitGPT with data streaming from Cloudflare R2 (S3-compatible storage). It supports both LitData and Parquet formats and is designed to run on a single node with 8 GPUs.

## Features

- ✅ Pretrain Qwen-2.5 1.5B model using LitGPT
- ✅ Stream training data directly from Cloudflare R2
- ✅ Support for both LitData and Parquet data formats
- ✅ Efficient multi-GPU training (8 GPUs)
- ✅ Data preparation and tokenization pipeline
- ✅ Automatic checkpoint saving and resumption
- ✅ TensorBoard logging support

## Project Structure

```
qwen_pretrain_project/
├── config/
│   ├── pretrain_qwen2.5_1.5b.yaml          # LitData format config
│   └── pretrain_qwen2.5_1.5b_parquet.yaml  # Parquet format config
├── pretrain_qwen_r2.py                     # Main pretraining script
├── r2_parquet_dataset.py                   # Parquet dataset loader
├── prepare_data_for_r2.py                  # Data preparation for LitData
├── run_pretrain.sh                         # Shell script to run training
├── test_r2_connection.py                   # R2 connection test
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

## Prerequisites

1. **Hardware**: Single node with 8 GPUs (e.g., 8x A100 or H100)
2. **Software**: Python 3.8+, CUDA 11.8+
3. **Cloudflare R2 Account**: With bucket and credentials

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set R2 Credentials

Export your R2 credentials as environment variables:

```bash
export R2_ACCESS_KEY_ID="your_access_key_id"
export R2_SECRET_ACCESS_KEY="your_secret_access_key"
export R2_ENDPOINT_URL="https://YOUR-ACCOUNT-ID.r2.cloudflarestorage.com"
```

### 3. Choose Data Format

This project supports two data formats:

#### Option A: LitData Format

LitData format is optimized for streaming tokenized data. Use this if you have raw text that needs tokenization.

##### Prepare Your Data

```bash
# For text files
python prepare_data_for_r2.py \
    --input-dir /path/to/raw/text/files \
    --output-dir ./optimized_data \
    --tokenizer Qwen/Qwen2.5-1.5B \
    --max-length 2048 \
    --file-type txt \
    --split train

# Upload to R2 (optional)
python prepare_data_for_r2.py \
    --input-dir /path/to/raw/files \
    --output-dir ./optimized_data \
    --upload-to-r2 \
    --r2-bucket your-bucket-name \
    --r2-prefix qwen-pretrain-data
```

##### Configure Training

Use `config/pretrain_qwen2.5_1.5b.yaml` and update the `data.path` with your R2 bucket path.

#### Option B: Parquet Format

Parquet format is ideal for pre-existing datasets stored as Parquet files with a "text" column.

##### Configure Training

Use `config/pretrain_qwen2.5_1.5b_parquet.yaml` and update:
- `data.path`: Your R2 bucket path to Parquet files
- `data.tokenizer`: Tokenizer to use

The Parquet dataset expects:
- Files with `_shard_sizes.json` and `_metadata.yaml` in the dataset root
- Parquet files with a "text" column containing raw text

### 4. Start Pretraining

For LitData format:
```bash
python pretrain_qwen_r2.py config/pretrain_qwen2.5_1.5b.yaml
```

For Parquet format:
```bash
python pretrain_qwen_r2.py config/pretrain_qwen2.5_1.5b_parquet.yaml
```

Or use the shell script:
```bash
chmod +x run_pretrain.sh
./run_pretrain.sh config/pretrain_qwen2.5_1.5b_parquet.yaml
```

## Data Format Details

### LitData Format
- Pre-tokenized data optimized for streaming
- Stored as binary chunks with metadata
- Best for custom datasets where you control tokenization

### Parquet Format
- Standard Parquet files with text data
- Tokenization happens on-the-fly during training
- Efficient caching and parallel processing
- Ideal for large datasets already in Parquet format

## Configuration Options

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `global_batch_size` | 512 | Total batch size across all GPUs |
| `micro_batch_size` | 64 | Batch size per GPU |
| `max_seq_length` | 2048 | Maximum sequence length |
| `max_tokens` | 100B | Total training tokens |
| `learning_rate` | 6e-4 | Peak learning rate |
| `lr_warmup_steps` | 2000 | Warmup steps |

### Data Configuration

For LitData:
```yaml
data: LitData
data.path: s3://your-bucket/path
```

For Parquet:
```yaml
data:
  format: parquet
  path: s3://your-bucket/path
  tokenizer: Qwen/Qwen2.5-1.5B
  num_workers: 8
  pack_samples: true
```

## Monitoring

Training progress is logged to TensorBoard:

```bash
tensorboard --logdir ./checkpoints/qwen2.5-1.5b-pretrain/logs
```

## Testing

Test R2 connection:
```bash
python test_r2_connection.py
```

Test data streaming:
```bash
python test_r2_connection.py s3://your-bucket/path/to/data
```

## Advanced Usage

### Custom Data Module

The project includes two data modules:
- `R2StreamingDataModule`: For LitData format
- `R2ParquetDataModule`: For Parquet format

Both can be customized in `pretrain_qwen_r2.py`.

### Using Different Models

To pretrain a different Qwen model size, change `model_name` in the config:

- `Qwen2.5-0.5B`
- `Qwen2.5-1.5B` (default)
- `Qwen2.5-3B`
- `Qwen2.5-7B`
- `Qwen2.5-14B`
- `Qwen2.5-32B`
- `Qwen2.5-72B`

### Performance Optimization

The Parquet loader includes several optimizations:
- Parallel file reading with thread pools
- Caching of Parquet file handles and metadata
- Batch tokenization for efficiency
- Worker-based sharding for distributed loading

## Troubleshooting

### Common Issues

1. **R2 Connection Error**: Check credentials and endpoint URL
2. **OOM Error**: Reduce `micro_batch_size` or `max_seq_length`
3. **Slow Data Loading**: 
   - For Parquet: Increase `num_workers` or check network bandwidth
   - For LitData: Ensure data is properly chunked
4. **Missing Metadata**: For Parquet format, ensure `_shard_sizes.json` and `_metadata.yaml` exist

### Performance Tips

1. Use `s5cmd` for faster S3 operations (automatically detected if installed)
2. For Parquet: The loader caches file handles and metadata for efficiency
3. Adjust `num_workers` based on your CPU count and network bandwidth
4. Enable `torch.compile` for faster training (PyTorch 2.0+)

## References

- [LitGPT Documentation](https://github.com/Lightning-AI/litgpt)
- [LitData Documentation](https://github.com/Lightning-AI/litdata)
- [Qwen Model Card](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- [Cloudflare R2 Documentation](https://developers.cloudflare.com/r2/)

## License

This project is released under the Apache 2.0 License.
