import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TextStreamer
import random
import csv
from tqdm import tqdm
import numpy as np
import ast
import sys
import time
import argparse
import pandas as pd
from utils.utils import get_last_processed_index, load_model_and_tokenizer


# -----------------------------
# Configurations and Parameters
# -----------------------------
# Set up argument parser
parser = argparse.ArgumentParser(description="Train a FastLanguageModel with specified parameters.")
parser.add_argument('--max_seq_length', type=int, default=40000, help='Maximum sequence length')
parser.add_argument('--max_attempts', type=int, default=6, help='Maximum sequence length')
parser.add_argument('--num_return_sequences', type=int, default=10, help='Maximum sequence length')
parser.add_argument('--max_new_tokens', type=int, default=5000, help='Maximum new tokens to generate')
parser.add_argument('--output_accord', action='store_true',default=False, help='Weather to train using Accord -> style output or not')
parser.add_argument('--output_list_of_lists', action='store_true',default=False, help='Weather to train using index and coord route as an output or not')
parser.add_argument('--use_distance', action='store_true',default=True, help='Weather to use paierd distance info in the input or not')
parser.add_argument('--model_1B', action='store_true',default=False, help='Weather to infer with 1B model or not')
parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'],  help='Data type (bfloat16 or float16)')
parser.add_argument('--load_in_4bit', action='store_true', default=True,  help='Use 4-bit quantization to reduce memory usage')
parser.add_argument('--infer_binpack', action='store_true',default=False, help='Weather to infer binpack model or not')

# Output directory
parser.add_argument('--output_dir', type=str, default=None, help='Output directory name')

args = parser.parse_args()

    
if args.infer_binpack:
    from feasibility_check_utils.binpack import validate_accord_format, validate_list_of_lists_format
else:
    from feasibility_check_utils.knapsack import validate_accord_format, validate_list_of_lists_format


# -----------------------------
# Load the Model and Tokenizer
# -----------------------------
if args.infer_binpack:
    model_dir = "binpack_models"
else:
    model_dir = "knapsack_models"

if args.output_accord:
    if args.model_1B:
        model_name = f"finetuned_models/{model_dir}/llama1B/accord/"
    else:
        model_name = f"finetuned_models/{model_dir}/llama8b/accord/"
elif args.output_list_of_lists:
    if args.model_1B:
        model_name = f"finetuned_models/{model_dir}/llama1b/list_of_lists/"
    else:
        model_name = f"finetuned_models/{model_dir}/llama8b/list_of_lists/"


print("model_name", model_name)

model, tokenizer, text_streamer = load_model_and_tokenizer(model_name, args.max_seq_length, args.dtype, args.load_in_4bit)


# -----------------------------
# Define the Alpaca Prompt Format
# -----------------------------
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]

    if args.output_accord:
        outputs = examples["output_accord"]
    elif args.output_list_of_lists:
        outputs = examples["output_list_of_lists"]

    texts = []
    for instruction, input_, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_, output)
        texts.append(text)
    return {"text": texts}



# -----------------------------
# Load the Evaluation Dataset
# -----------------------------

if args.infer_binpack:
    eval_dataset = load_dataset("json", data_files='validation_data/binpack_val_data.json')
else:
    # eval_dataset = load_dataset("json", data_files="train_data/knapsack_train_data/knapsak_train_data.json")

    # eval_dataset = load_dataset("json", data_files='validation_data/knapsak_val_data.json')
    eval_dataset = load_dataset("json", data_files='validation_data/filtered_items_knapsak.json')

# Apply the formatting function to the eval dataset
eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

# -----------------------------
# Evaluation Loop: Generate and Validate Solutions
# -----------------------------

# -----------------------------
# Evaluation Loop: Generate and Validate Solutions
# -----------------------------
def evaluate_model(
    model,
    tokenizer,
    dataset,
    csv_filename="evaluation_results.csv",
    max_attempts=5,
    num_return_sequences=1,
):
    all_predictions = []
    all_references = []

    # Get the last processed index
    last_processed_index = get_last_processed_index(csv_filename)
    start_idx = last_processed_index + 1
    
    # Check if the file already exists to determine if we need to write the header
    write_header = not os.path.exists(csv_filename)
    
    # Open CSV file in append mode to continue from where we left off
    file_mode = "w" if write_header else "a"
    # Write CSV header.
    with open(csv_filename, mode=file_mode, newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(
                [   "Index",
                    "Num_Items",         # Number of items in the instance.
                    "Capacity",          # Knapsack capacity.
                    "Declared_Value",    # Value declared by the LLM.
                    "Optimal_Value",     # Optimal computed value.
                    "Gap",               # (Optimal_Value - Declared_Value).
                    "Total_Sampling_Attempts",
                    "Num_Feasible_Solutions",
                    "Feasible_Solution_Values",  # List of declared values from feasible solutions.
                    "Validation_Message",
                    "Path",
                    "Best_Solution",
                    "Avg_Time"  
                ]
                
            )
    real_feas_list = []
    gap_list = []
    bad_items   = []
    real_optimal_value_equal_list = []
    print("start_idx", start_idx)
    for idx, example in tqdm(enumerate(dataset["train"])):
        # Skip examples that have already been processed
        # print(list(example.keys()))
        if idx < start_idx:
            continue
        # Example fields: "instruction", "input", "output", "num_items", "capacities", "path", etc.


        if args.infer_binpack:
            num_items = example.get("num_items")
            capacity = example.get("bin_capacity")
            real_optimal_value = example.get("num_bins_sol")
            input_items       = example.get("input")
        else:
            input_items = example["input"]
            capacity = example.get("capacities", [20])[0]  # Get the first capacity if it's a list
            num_items = example.get("num_items", len(input_items))
            real_optimal_value = example.get("value", None)

        
        print("input_items", input_items)
        # id = example["id"]
        # print("num_items", num_items, flush=True)
        # print("capacity", capacity, flush=True)
        # print("input_items", input_items, flush=True)
        if args.output_accord:
            real_solution =  example.get("output_accord")
            feasibility_checker = validate_accord_format
        elif args.output_list_of_lists:
            real_solution =  example.get("output_list_of_lists")
            feasibility_checker = validate_list_of_lists_format

        print(f"Processing example {idx}")
        print("num_items ", num_items)
        

        
        # Create prompt without revealing the expected output.
        prompt = alpaca_prompt.format(example["instruction"], input_items, "")


       
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to("cuda")

        feasible_solutions = []
        feasible_values = []
        feasible_validation_messages = []
        real_optimal_value_list = []
        feasible_solution_gaps = []
        num_feasible_solutions = 0
        total_sampling_attempts = 0
        infeasible_validation_message = ""
        total_time = 0.0  # Initialize total time for this instance

        # Generate multiple attempts.
        found_threshold = False  # Flag to stop once gap is within the threshold.
        for attempt in range(max_attempts):
            total_sampling_attempts += num_return_sequences
            seed = random.randint(0, 1000000)
            torch.manual_seed(seed)

            # Start timing before generation.
            start_time = time.time()
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
            # #for debugging
            # generated_texts = [real_solution]
            # End timing after generation.
            end_time = time.time()
            total_time += (end_time - start_time)

            for k, gen_text in enumerate(generated_texts):
                
                is_feasible, message, computed_value, real_optimal_value_ret = feasibility_checker(input_items, gen_text, capacity, real_optimal_value)
                real_optimal_value_equal_list.append(real_optimal_value_ret == real_optimal_value)
                print("real_optimal_value == real_optimal_value returned", sum(real_optimal_value_equal_list)/len(real_optimal_value_equal_list))

                if(computed_value is None) or (real_optimal_value is None):
                    print("computed_value and real_optimal_value is None")
                    continue

                if is_feasible:
                    
                    feasible_solutions.append(gen_text)
                    feasible_values.append(computed_value)
                    feasible_validation_messages.append(message)
                    num_feasible_solutions += 1


                    gap = abs((real_optimal_value - computed_value) / real_optimal_value)

                    feasible_solution_gaps.append(gap)
                    real_optimal_value_list.append(real_optimal_value)

                    print("num_items ", num_items, "calculated_gap : ", gap)
                    #for debugging
                    real_feas_list.append(is_feasible)
                    gap_list.append(gap)
                    print("average of is_feasible : ", sum(real_feas_list)/len(real_feas_list))
                    print('sum(gap_list) ', sum(gap_list))

                    if gap <= 0.05:
                        found_threshold = True
                        break
                else:
                    infeasible_validation_message = message

            # Exit the outer generation loop if a sufficient solution is found.
            if found_threshold:
                 break

        
        # sys.exit()

        if feasible_solutions:
            # Choose the best solution (here using the highest declared value).
            best_index = np.argmin(feasible_solution_gaps)
            best_solution = feasible_solutions[best_index]
            best_declared_value = feasible_values[best_index]
            best_message = feasible_validation_messages[best_index]
            best_gap = feasible_solution_gaps[best_index]
            best_real_optimal_value = real_optimal_value_list[best_index]
        else:
            best_solution = ""
            best_declared_value = None
            best_real_optimal_value = None
            best_gap = None
            best_message = infeasible_validation_message

        all_predictions.append(best_solution)
        # all_references.append([example["output"]])
        path = example.get("path", "")
        # Calculate the average time per attempt.
        avg_time = total_time / (max_attempts*num_return_sequences) if max_attempts > 0 else 0.0


        print("bad_items", bad_items)

        # Write results for the current instance.
        with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [   idx,
                    num_items,
                    capacity,
                    best_declared_value,
                    best_real_optimal_value,
                    best_gap,
                    total_sampling_attempts,
                    num_feasible_solutions,
                    feasible_values,
                    best_message,
                    path,
                    best_solution,
                    # Calculate the average time per attempt.
                    avg_time
                ]
            )

    return all_predictions, all_references


# -----------------------------
# Main Execution
# -----------------------------
print("Starting evaluation loop...")
import os
model_base_name = os.path.basename(model_name) if model_name else "model"

if args.infer_binpack:
    output_dir = "val_results/binpack_val"

    if args.model_1B:
        output_dir = "val_results/binpack_val_1B"
else:
    output_dir = "val_results/knapsack_val"
    if args.model_1B:
        output_dir = "val_results/knapsack_val_1B"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


if args.use_distance and args.output_accord:
    csv_filename = f"{output_dir}/output_accord_{model_base_name}_num_return_sequences{args.num_return_sequences}.csv"
elif args.use_distance and args.output_list_of_lists:
    csv_filename = f"{output_dir}/output_list_of_lists_{model_base_name}_num_return_sequences{args.num_return_sequences}.csv"



predictions = evaluate_model(
    model,
    tokenizer,
    eval_dataset,
    max_attempts=args.max_attempts,
    num_return_sequences=args.num_return_sequences,
    csv_filename=csv_filename,
)

print("Evaluation complete. Results saved to:", csv_filename)
