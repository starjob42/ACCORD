import argparse
import torch
import wandb
from unsloth import FastLanguageModel
from datasets import load_dataset
from unsloth import is_bfloat16_supported
import csv
import os
import seaborn as sns
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import pandas as pd
from pathlib import Path
import json

torch.cuda.empty_cache()
torch.cuda.ipc_collect()


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a FastLanguageModel with specified parameters.")
    
    # Model and data parameters
    parser.add_argument('--max_seq_length', type=int, default=40000, help='Maximum sequence length')
    parser.add_argument('--model_type', type=str, default='llama8b', choices=['llama8b', 'llama1b'], 
                        help='Which model to use')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'], 
                        help='Data type (bfloat16 or float16)')
    parser.add_argument('--load_in_4bit', action='store_true', default=True, 
                        help='Use 4-bit quantization to reduce memory usage')

    # LoRA hyperparameters
    parser.add_argument('--lora_r', type=int, default=64, help='Rank of the LoRA decomposition')
    parser.add_argument('--lora_alpha', type=int, default=64, help='Scaling factor for LoRA updates')
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='Dropout rate for LoRA layers')
    parser.add_argument('--bias', type=str, default='none', choices=['none', 'all', 'lora_only'], help='Bias type')

    # Additional configurations
    parser.add_argument('--use_gradient_checkpointing', type=str, default='unsloth', help='Use gradient checkpointing')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--use_rslora', action='store_true', default=True, help='Use RSLoRA')
    parser.add_argument('--loftq_config', type=str, default=None, help='LoFT-Q configuration')

    # Training hyperparameters
    parser.add_argument('--per_device_train_batch_size', type=int, default=8, 
                        help='Batch size per device during training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, 
                        help='Number of gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=10, help='Number of warmup steps')
    parser.add_argument('--num_train_epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--embedding_learning_rate', type=float, default=1e-5, help='Embedding learning rate')
    parser.add_argument('--logging_steps', type=int, default=1, help='Logging steps')
    parser.add_argument('--optim', type=str, default='adamw_8bit', help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='Learning rate scheduler type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_total_limit', type=int, default=100, help='Total save limit for model checkpoints')
    parser.add_argument('--save_step', type=int, default=5, help='Steps interval to save model checkpoints')
    parser.add_argument('--eval_steps', type=int, default=50, help='Steps interval to save model checkpoints')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1, 
                        help='Batch size per device during evaluation')
    
    # Model training options
    parser.add_argument('--train_lm_head', action='store_true', default=False, 
                        help='Whether to train the language model head or not')
    parser.add_argument('--train_embed_tokens', action='store_true', default=False, 
                        help='Whether to train the embed_tokens or not')
    
    # Output format options
    parser.add_argument('--output_accord', action='store_true', default=False, 
                        help='Whether to train using ACCORD-style output or not')
    parser.add_argument('--output_list_of_lists', action='store_true', default=False, 
                        help='Whether to train using only list of lists of index route as an output or not')
    
    # Task type options
    parser.add_argument('--train_vrp_tsp', action='store_true', default=False, 
                        help='Whether to train VRP-TSP model or not')
    parser.add_argument('--train_knapsack', action='store_true', default=False, 
                        help='Whether to train KNAPSACK model or not')
    parser.add_argument('--train_binpack', action='store_true', default=False, 
                        help='Whether to train BINPACK model or not')
    parser.add_argument('--train_jssp', action='store_true', default=False, 
                        help='Whether to train JSSP model or not')
    parser.add_argument('--train_fssp', action='store_true', default=False, 
                        help='Whether to train FSSP model or not')
    
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory name')
    parser.add_argument('--wandb_project', type=str, default=None, 
                        help='WandB project name (default: derived from task type)')

    args = parser.parse_args()

    # Print output style information
    if args.output_accord:
        print("=="*60)
        print("Training with ACCORD style output")
        print("=="*60)
    else:
        print("=="*60)
        print("Training with list of lists style output")
        print("=="*60)
    
    # Determine the task type
    task_type = None
    if args.train_vrp_tsp:
        task_type = "vrp_tsp"
    elif args.train_knapsack:
        task_type = "knapsack"
    elif args.train_binpack:
        task_type = "binpack"
    elif args.train_jssp:
        task_type = "jssp"
    elif args.train_fssp:
        task_type = "fssp"
    else:
        raise ValueError("No task type selected. Please specify a training task.")

    # =========================
    # Load Model and Tokenizer
    # =========================

    # Correct the model names
    if args.model_type == 'llama8b':
        model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    elif args.model_type == 'llama1b':
        model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    base_model = os.path.basename(model_name)
    print("base_model: ", base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16,
        local_files_only=False,
    ) 
    
    print("Model loaded successfully.")
    print("Model dtype: ", model.dtype)
    print(f"args.max_seq_length {args.max_seq_length}")
    print("Model max_seq_length: ", model.config.max_position_embeddings)

    # Define modules to train
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", 
    ]
    if args.train_lm_head:
        target_modules.append("lm_head")
    if args.train_embed_tokens:
        target_modules.append("embed_tokens")

    print("Target modules: ", target_modules)

    # Configure the model with PEFT (Parameter-Efficient Fine-Tuning)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        use_rslora=args.use_rslora,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.random_state,
        loftq_config=args.loftq_config,
    )

    print(model.print_trainable_parameters())

    # Define the Alpaca-style prompt template
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    def formatting_prompts_st(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]

        #adding paired city distances to the input
        if args.train_vrp_tsp:
            k = 0
            for i, j in zip(inputs, examples["paired_distances"]):
                inputs[k] = "coordinates: " + i + "\n" + "distances : " + j
                k += 1
                                                 
        if args.output_accord:
            outputs = examples["output_accord"]
        elif args.output_list_of_lists:
            outputs = examples["output_list_of_lists"]
        
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}
    
    # Load the dataset based on task type
    dataset_path = f"train_data/{task_type}_train_data/"
    dataset = load_dataset(dataset_path, split="train")

    formatting_prompts_func = formatting_prompts_st
    split_dataset = dataset.train_test_split(test_size=0.05, seed=args.seed)

    train_dataset = split_dataset['train'].map(formatting_prompts_func, batched=True)
    eval_dataset = split_dataset['test'].map(formatting_prompts_func, batched=True)

    print("train_dataset length : ", len(train_dataset))
    print("eval_dataset length : ", len(eval_dataset))

    print("train_dataset: ", train_dataset[0])
    print("train_dataset: ", eval_dataset[0])

    
    # # Analyze token lengths
    # texts = train_dataset["text"]
    # tokenized_lengths = [len(tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]) for text in texts]

    # # Calculate statistics
    # avg_length = sum(tokenized_lengths) / len(tokenized_lengths)
    # min_length = min(tokenized_lengths)
    # max_length = max(tokenized_lengths)

    # print("Average prompt length:", avg_length)
    # print("Minimum prompt length:", min_length)
    # print("Maximum prompt length:", max_length)

    # =========================
    # Generate Output Directory Name
    # =========================
    
    # Create a concise but descriptive output directory name
    output_style = "accord" if args.output_accord else "list_of_lists" if args.output_list_of_lists else "default"
    
    if args.output_dir is None:
        # Create a clean and informative directory name
        model_short_name = "llama8b" if "8B" in base_model else "llama1b" if "1B" in base_model else base_model.replace("-", "_")
        
        dir_out = os.path.join(
            "finetuned_models",
            f"{task_type}_{model_short_name}_{output_style}_r{args.lora_r}_ep{args.num_train_epochs}"
        )
    else:
        dir_out = args.output_dir

    os.makedirs(dir_out, exist_ok=True)
    print("Output directory: ", dir_out)
    
    # =========================
    # Initialize WandB
    # =========================
    
    # Create a descriptive wandb run name
    wandb_run_name = f"{task_type}_{model_short_name}_{output_style}_r{args.lora_r}"
    
    # Set the project name based on task or user preference
    if args.wandb_project:
        project_name = args.wandb_project
    else:
        project_name = f"{task_type}_optimization"
    
    wandb.init(
        project=project_name,
        name=wandb_run_name,
        config=vars(args)  # Log all args to wandb
    )

    # Save hyperparameters to CSV
    with open(os.path.join(dir_out, 'training_hyperparams_args.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in vars(args).items():
            writer.writerow([key, value])

    # =========================
    # Initialize the Trainer
    # =========================

    if args.train_lm_head and args.train_embed_tokens:
        from unsloth import UnslothTrainer, UnslothTrainingArguments
        Trainer = UnslothTrainer
        TrainingArguments = UnslothTrainingArguments
        print("Training with UnslothTrainer")
    else:
        from trl import SFTTrainer
        from transformers import TrainingArguments
        Trainer = SFTTrainer
        TrainingArguments = TrainingArguments
        print("Training with SFTTrainer")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=40,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            bf16=True if args.dtype == "bfloat16" else False,
            fp16=True if args.dtype == "float16" else False,
            logging_steps=args.logging_steps,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            seed=args.seed,
            output_dir=dir_out,
            report_to="wandb",
            load_best_model_at_end=False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=args.save_total_limit,
            save_steps=args.save_step,
            evaluation_strategy="steps",
            eval_steps=args.eval_steps,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
        ),
    )

    # =========================
    # Monitor GPU Memory Usage
    # =========================
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # =========================
    # Start Training
    # =========================
    contains_checkpoints = any(
        "checkpoint" in name for name in os.listdir(dir_out) if os.path.isdir(os.path.join(dir_out, name))
    )

    if not contains_checkpoints:
        print("Checkpoint dir is empty: Training new model")
        trainer.train()
    else:
        print("Checkpoint dir is NOT empty: Continuing training")
        trainer.train(resume_from_checkpoint=True)

if __name__ == "__main__":
    main()