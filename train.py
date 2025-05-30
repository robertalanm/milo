#!/usr/bin/env python3
"""
Pretraining script using HuggingFace Accelerate and StreamingDataset
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain a transformers model with streaming dataset")
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="c4",
        help="The name of the dataset to use (via HuggingFace datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="en",
        help="The configuration name of the dataset to use (via HuggingFace datasets library).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming mode for the dataset",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes, truncate the number of training examples.",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt2",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="Train model from scratch instead of from pretrained weights",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    
    # Training arguments
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Ratio of total training steps used for warmup.",
    )
    
    # Data processing arguments
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="The maximum total sequence length for target text after tokenization.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training sets",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss",
    )
    
    # Other arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="1000",
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="The integration to report the results and logs to. Supported platforms are `tensorboard`,"
        " `wandb`, `comet_ml` and `clearml`. Use `all` (default) to report to all integrations.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.",
    )
    
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    
    # Store whether we adjusted learning rate for logging later
    adjusted_lr = False
    if args.from_scratch and args.learning_rate == 5e-5:
        args.learning_rate = 1e-4
        adjusted_lr = True
    
    # Initialize the accelerator
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )
    
    # Make one log on every process with the configuration for debugging
    logger.info(accelerator.state)
    
    # Setup logging
    if accelerator.is_local_main_process:
        logger.setLevel("INFO")
    else:
        logger.setLevel("ERROR")
    
    # Log learning rate adjustment after accelerator is initialized
    if adjusted_lr:
        logger.info(f"Training from scratch: adjusted learning rate to {args.learning_rate}")
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()
    
    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=True,
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=True,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
        )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    logger.info(f"Loading dataset {args.dataset_name}")
    if args.streaming:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            streaming=True,
            split="train",
        )
    else:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split="train",
        )
    
    # Preprocessing function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_seq_length,
            return_special_tokens_mask=True,
        )
    
    # Tokenize dataset
    logger.info("Tokenizing dataset")
    if args.streaming:
        # Get column names from the first batch for streaming datasets
        # We need to peek at the dataset to get column names
        first_batch = next(iter(dataset.take(1)))
        column_names = list(first_batch.keys())
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )
    else:
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    
    # Limit samples if specified
    if args.max_train_samples is not None and not args.streaming:
        max_train_samples = min(len(tokenized_dataset), args.max_train_samples)
        tokenized_dataset = tokenized_dataset.select(range(max_train_samples))
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing CLM not MLM
        pad_to_multiple_of=8,
    )
    
    # DataLoader
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=False if args.streaming else True,
    )
    
    # Load model
    logger.info(f"Loading model {args.model_name_or_path}")
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=True)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    else:
        raise ValueError("You must provide either a model name or a config name")
    
    if args.from_scratch:
        logger.info("Initializing model from scratch with random weights")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        
        # Initialize weights properly for training from scratch
        def _init_weights(module):
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if isinstance(module, torch.nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        
        model.apply(_init_weights)
        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=config,
            trust_remote_code=True,
        )
    
    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Optimizer
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Scheduler and math around the number of training steps
    overrode_max_train_steps = False
    if args.max_train_steps is None:
        if args.streaming:
            args.max_train_steps = args.num_train_epochs * 1000000  # Arbitrary large number
            overrode_max_train_steps = True
        else:
            args.max_train_steps = args.num_train_epochs * len(train_dataloader)
            overrode_max_train_steps = True
    
    # Calculate warmup steps
    if args.num_warmup_steps == 0 and args.warmup_ratio > 0:
        args.num_warmup_steps = int(args.max_train_steps * args.warmup_ratio)
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Recalculate total training steps as the size of the dataloader may have changed
    if overrode_max_train_steps and not args.streaming:
        args.max_train_steps = args.num_train_epochs * len(train_dataloader)
    
    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)
    
    # Initialize trackers
    if args.with_tracking:
        logger.info("Initializing trackers...")
        experiment_config = vars(args)
        accelerator.init_trackers("milo-one-3b-training", experiment_config)

        # if args.report_to == "wandb":
        #     logger.info("Initializing wandb trackers...")
        #     accelerator.init_trackers(
        #         "milo-one-3b-training",
        #         config=vars(args),
        #         init_kwargs={"wandb": {"entity": "0xcarro"}}
        #     )
    
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(tokenized_dataset) if not args.streaming else 'streaming'}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Only show the progress bar once on each machine
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(checkpoint_path)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(args.output_dir) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1] if len(dirs) > 0 else None
            checkpoint_path = os.path.join(args.output_dir, path)
        
        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]
        
        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * len(train_dataloader)
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)
    
    # Update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    
    # Training loop
    model.train()
    total_loss = 0
    train_losses = []
    
    logger.info("Starting training loop...")
    
    # Track training start time for tokens/sec calculation
    training_start_time = time.time()
    
    for epoch in range(starting_epoch, args.num_train_epochs):
        logger.info(f"Starting epoch {epoch + 1} of {args.num_train_epochs}")
        
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # Skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                
                # Log every step to progress bar
                current_loss = loss.detach().float().item()
                
                # Calculate tokens per second
                elapsed_time = time.time() - training_start_time
                tokens_processed = total_batch_size * args.max_seq_length * completed_steps
                tokens_per_sec = tokens_processed / elapsed_time if elapsed_time > 0 else 0
                
                progress_bar.set_postfix({
                    'loss': f'{current_loss:.4f}', 
                    'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}',
                    'tokens/s': f'{tokens_per_sec:.0f}'
                })
                
                # Logging
                if completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / args.logging_steps / args.gradient_accumulation_steps
                    train_losses.append(avg_loss)
                    logger.info(f"Step: {completed_steps}, Loss: {avg_loss:.4f}, LR: {lr_scheduler.get_last_lr()[0]:.2e}, Tokens/sec: {tokens_per_sec:.0f}")
                    
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "train_loss": avg_loss,
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "epoch": epoch,
                                "step": completed_steps,
                                "tokens_per_second": tokens_per_sec,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0
                
                # Checkpointing
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                        
                        # Save model
                        if accelerator.is_main_process:
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(
                                output_dir,
                                is_main_process=accelerator.is_main_process,
                                save_function=accelerator.save,
                            )
                            tokenizer.save_pretrained(output_dir)
            
            if completed_steps >= args.max_train_steps:
                break
        
        # Checkpointing per epoch
        if checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            
            # Save model
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )
                tokenizer.save_pretrained(output_dir)
        
        if completed_steps >= args.max_train_steps:
            break
    
    # End training
    if args.with_tracking:
        accelerator.end_training()
    
    # Save final model
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            
            # Save training args
            with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
            
            # Save training losses
            with open(os.path.join(args.output_dir, "train_losses.json"), "w") as f:
                json.dump(train_losses, f)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main() 