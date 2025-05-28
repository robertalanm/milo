#!/bin/bash
# Script to run Qwen-2.5 1.5B pretraining on 8 GPUs with R2 streaming

# Exit on error
set -e

# Check if required environment variables are set
if [ -z "$R2_ACCESS_KEY_ID" ] || [ -z "$R2_SECRET_ACCESS_KEY" ] || [ -z "$R2_ENDPOINT_URL" ]; then
    echo "Error: R2 credentials not set!"
    echo "Please set the following environment variables:"
    echo "  export R2_ACCESS_KEY_ID=your_access_key"
    echo "  export R2_SECRET_ACCESS_KEY=your_secret_key"
    echo "  export R2_ENDPOINT_URL=https://YOUR-ACCOUNT-ID.r2.cloudflarestorage.com"
    exit 1
fi

# Configuration
CONFIG_FILE="${1:-config/pretrain_qwen2.5_1.5b.yaml}"
NUM_GPUS=8

echo "Starting Qwen-2.5 1.5B pretraining with configuration: $CONFIG_FILE"
echo "Using $NUM_GPUS GPUs"

# Install dependencies if requirements.txt has changed
if [ -f "requirements.txt" ]; then
    echo "Checking dependencies..."
    pip install -q -r requirements.txt
fi

# Method 1: Using the custom script with R2 support (RECOMMENDED)
echo "Running custom pretraining script with R2 streaming support..."
# python pretrain_qwen_r2.py $CONFIG_FILE

# Alternative Method 2: Using LitGPT CLI directly
# Note: This only works with standard LitGPT data modules
# Uncomment below if you want to use the standard LitGPT CLI
#
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# litgpt pretrain \
#     --config $CONFIG_FILE \
#     --devices $NUM_GPUS \
#     --num_nodes 1

# Alternative Method 3: Using torchrun for distributed training
# Uncomment below for explicit distributed training control
#
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    pretrain_qwen_r2.py $CONFIG_FILE 