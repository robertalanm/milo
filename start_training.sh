#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}    HuggingFace Pretraining Script    ${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to display usage
usage() {
    echo -e "${YELLOW}Usage: $0 [OPTIONS]${NC}"
    echo "Options:"
    echo "  -m, --mode          Training mode: 'single' (single GPU), 'multi' (multi-GPU), 'stream' (streaming dataset)"
    echo "  -d, --dataset       Dataset name (default: c4)"
    echo "  -n, --model         Model name or path (default: gpt2)"
    echo "  -b, --batch-size    Batch size per device (default: 8)"
    echo "  -s, --steps         Max training steps (default: 10000)"
    echo "  -l, --lr            Learning rate (default: 5e-5)"
    echo "  -o, --output        Output directory (default: ./output)"
    echo "  -r, --resume        Resume from checkpoint path"
    echo "  -t, --tracking      Enable experiment tracking (wandb/tensorboard)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --mode single --dataset c4 --model gpt2 --batch-size 8"
    echo "  $0 --mode multi --steps 50000 --tracking"
    echo "  $0 --mode stream --dataset c4 --model gpt2-medium"
    exit 1
}

# Default values
MODE="multi"
DATASET="mlfoundations/dclm-baseline-1.0"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
BATCH_SIZE=8
MAX_STEPS=10000
LEARNING_RATE=5e-5
OUTPUT_DIR="./output"
RESUME=""
TRACKING=false
DATASET_CONFIG="en"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -n|--model)
            MODEL="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -s|--steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        -l|--lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME="$2"
            shift 2
            ;;
        -t|--tracking)
            TRACKING=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Check if virtual environment exists
if [ ! -d "~/.venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv ~/.venv
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source ~/.venv/bin/activate

# Install dependencies if needed
if ! pip show accelerate &> /dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install uv
    uv pip install --upgrade pip
    uv pip install -r requirements.txt
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build the base command
BASE_CMD="python train.py \
    --dataset_name $DATASET \
    --dataset_config_name $DATASET_CONFIG \
    --model_name_or_path $MODEL \
    --per_device_train_batch_size $BATCH_SIZE \
    --max_train_steps $MAX_STEPS \
    --learning_rate $LEARNING_RATE \
    --output_dir $OUTPUT_DIR \
    --logging_steps 100 \
    --checkpointing_steps 1000 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --max_seq_length 1024"

# Add tracking if enabled
if [ "$TRACKING" = true ]; then
    BASE_CMD="$BASE_CMD --with_tracking"
fi

# Add resume if specified
if [ -n "$RESUME" ]; then
    BASE_CMD="$BASE_CMD --resume_from_checkpoint $RESUME"
fi

# Execute based on mode
case $MODE in
    single)
        echo -e "${GREEN}Starting single GPU training...${NC}"
        echo -e "${YELLOW}Dataset: $DATASET${NC}"
        echo -e "${YELLOW}Model: $MODEL${NC}"
        echo -e "${YELLOW}Batch size: $BATCH_SIZE${NC}"
        echo -e "${YELLOW}Max steps: $MAX_STEPS${NC}"
        echo -e "${YELLOW}Learning rate: $LEARNING_RATE${NC}"
        echo -e "${YELLOW}Output dir: $OUTPUT_DIR${NC}"
        echo ""
        
        # Single GPU training
        CUDA_VISIBLE_DEVICES=0 $BASE_CMD
        ;;
        
    multi)
        echo -e "${GREEN}Starting multi-GPU training...${NC}"
        echo -e "${YELLOW}Dataset: $DATASET${NC}"
        echo -e "${YELLOW}Model: $MODEL${NC}"
        echo -e "${YELLOW}Batch size: $BATCH_SIZE per GPU${NC}"
        echo -e "${YELLOW}Max steps: $MAX_STEPS${NC}"
        echo -e "${YELLOW}Learning rate: $LEARNING_RATE${NC}"
        echo -e "${YELLOW}Output dir: $OUTPUT_DIR${NC}"
        
        # Check if accelerate config exists
        if [ ! -f "accelerate_config.yaml" ]; then
            echo -e "${YELLOW}No accelerate config found. Running accelerate config...${NC}"
            accelerate config
        fi
        
        # Multi-GPU training with accelerate
        accelerate launch --config_file accelerate_config.yaml $BASE_CMD
        ;;
        
    stream)
        echo -e "${GREEN}Starting training with streaming dataset...${NC}"
        echo -e "${YELLOW}Dataset: $DATASET (streaming)${NC}"
        echo -e "${YELLOW}Model: $MODEL${NC}"
        echo -e "${YELLOW}Batch size: $BATCH_SIZE${NC}"
        echo -e "${YELLOW}Max steps: $MAX_STEPS${NC}"
        echo -e "${YELLOW}Learning rate: $LEARNING_RATE${NC}"
        echo -e "${YELLOW}Output dir: $OUTPUT_DIR${NC}"
        echo ""
        
        # Streaming dataset training
        $BASE_CMD --streaming
        ;;
        
    *)
        echo -e "${RED}Invalid mode: $MODE${NC}"
        echo "Valid modes are: single, multi, stream"
        exit 1
        ;;
esac

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
else
    echo -e "${RED}Training failed with error code $?${NC}"
    exit 1
fi 