# HuggingFace Pretraining with Accelerate and Streaming Datasets

This repository contains a comprehensive pretraining script using HuggingFace's Accelerate library with support for streaming datasets, distributed training, and checkpoint resumption.

## Features

- ✅ **Distributed Training**: Support for single-GPU and multi-GPU training using HuggingFace Accelerate
- ✅ **Streaming Datasets**: Efficient training on large datasets without loading everything into memory
- ✅ **Checkpoint Resumption**: Continue training from saved checkpoints
- ✅ **Experiment Tracking**: Integration with Weights & Biases, TensorBoard, and other trackers
- ✅ **Flexible Configuration**: Extensive command-line arguments for customization
- ✅ **Memory Efficient**: Gradient accumulation and mixed precision training support

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-name>

# Make the training script executable
chmod +x start_training.sh
```

### 2. Install Dependencies

The training script will automatically create a virtual environment and install dependencies, but you can also do it manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Start Training

#### Single GPU Training
```bash
./start_training.sh --mode single --dataset c4 --model gpt2 --batch-size 8 --steps 10000
```

#### Multi-GPU Training
```bash
./start_training.sh --mode multi --dataset c4 --model gpt2 --batch-size 8 --steps 50000
```

#### Streaming Dataset Training
```bash
./start_training.sh --mode stream --dataset c4 --model gpt2-medium --steps 100000
```

## Usage

### Training Script Options

The `start_training.sh` script supports the following options:

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--mode` | `-m` | Training mode: 'single', 'multi', or 'stream' | single |
| `--dataset` | `-d` | Dataset name from HuggingFace Hub | c4 |
| `--model` | `-n` | Model name or path | gpt2 |
| `--batch-size` | `-b` | Batch size per device | 8 |
| `--steps` | `-s` | Maximum training steps | 10000 |
| `--lr` | `-l` | Learning rate | 5e-5 |
| `--output` | `-o` | Output directory | ./output |
| `--resume` | `-r` | Resume from checkpoint path | None |
| `--tracking` | `-t` | Enable experiment tracking | False |
| `--help` | `-h` | Show help message | - |

### Direct Python Usage

You can also run the training script directly with Python:

```bash
# Basic training
python train.py \
    --dataset_name c4 \
    --model_name_or_path gpt2 \
    --per_device_train_batch_size 8 \
    --max_train_steps 10000 \
    --output_dir ./output

# Streaming mode
python train.py \
    --dataset_name c4 \
    --model_name_or_path gpt2 \
    --streaming \
    --per_device_train_batch_size 4 \
    --max_train_steps 50000

# With experiment tracking
python train.py \
    --dataset_name openwebtext \
    --model_name_or_path gpt2-medium \
    --with_tracking \
    --report_to wandb \
    --max_train_steps 100000
```

### Multi-GPU Training with Accelerate

For distributed training across multiple GPUs:

```bash
# Configure accelerate (interactive)
accelerate config

# Or use the provided config
accelerate launch --config_file accelerate_config.yaml train.py \
    --dataset_name c4 \
    --model_name_or_path gpt2 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4
```

## Configuration Files

### `accelerate_config.yaml`

The provided configuration file is set up for multi-GPU training. Key settings:

- `distributed_type`: MULTI_GPU (change to NO for single GPU)
- `mixed_precision`: fp16 (can be 'no', 'fp16', or 'bf16')
- `num_processes`: Number of GPUs to use

### Model Configuration

You can specify a custom model configuration:

```bash
python train.py \
    --config_name path/to/config.json \
    --tokenizer_name gpt2 \
    --model_name_or_path gpt2
```

## Datasets

The script supports any dataset from the HuggingFace Hub that has a "text" column. Popular options:

- `c4` - Colossal Clean Crawled Corpus
- `openwebtext` - OpenWebText dataset
- `wikitext` - Wikipedia articles
- `bookcorpus` - Book corpus
- Custom datasets with a "text" field

### Using Custom Datasets

```bash
python train.py \
    --dataset_name your-username/your-dataset \
    --dataset_config_name your-config \
    --streaming
```

## Checkpointing and Resumption

The script automatically saves checkpoints during training:

```bash
# Resume from the latest checkpoint
./start_training.sh --resume output/step_5000

# Resume from a specific checkpoint
python train.py --resume_from_checkpoint output/step_5000
```

Checkpoints include:
- Model weights
- Optimizer state
- Learning rate scheduler state
- Random number generator states
- Training progress

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir output/

# View at http://localhost:6006
```

### Weights & Biases

```bash
# Login to W&B
wandb login

# Run with W&B tracking
./start_training.sh --mode single --tracking --steps 10000
```

## Performance Tips

1. **Gradient Accumulation**: For larger effective batch sizes
   ```bash
   python train.py --gradient_accumulation_steps 8 --per_device_train_batch_size 4
   ```

2. **Mixed Precision Training**: Already configured in `accelerate_config.yaml`

3. **Streaming Datasets**: Essential for large datasets
   ```bash
   ./start_training.sh --mode stream --dataset c4
   ```

4. **Data Loading**: Adjust number of workers
   ```bash
   python train.py --preprocessing_num_workers 8
   ```

## Troubleshooting

### CUDA Out of Memory

- Reduce `--per_device_train_batch_size`
- Increase `--gradient_accumulation_steps`
- Reduce `--max_seq_length`
- Enable gradient checkpointing (if supported by model)

### Slow Training

- Enable mixed precision training in `accelerate_config.yaml`
- Use streaming datasets for large datasets
- Increase `--preprocessing_num_workers`
- Check GPU utilization with `nvidia-smi`

### Connection Issues

For datasets requiring authentication:
```bash
huggingface-cli login
```

## Advanced Usage

### Custom Training Loop Modifications

The training script (`train.py`) can be easily modified for:
- Custom loss functions
- Different model architectures
- Special preprocessing
- Custom metrics

### Distributed Training on Multiple Nodes

Create a custom accelerate config:
```bash
accelerate config --config_file multi_node_config.yaml
```

Then launch on all nodes:
```bash
accelerate launch --config_file multi_node_config.yaml train.py
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is provided as-is for educational and research purposes.
