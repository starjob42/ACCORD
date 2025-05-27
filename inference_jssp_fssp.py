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
from utils.utils import get_last_processed_index, load_model_and_tokenizer, max_new_token_pred

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
parser.add_argument('--infer_fssp', action='store_true', default=False,  help='Infer flowshop scheduling problem or not')

# Output directory
parser.add_argument('--output_dir', type=str, default=None, help='Output directory name')

args = parser.parse_args()

    
if args.infer_fssp:
    from feasibility_check_utils.fssp import validate_accord_format, validate_list_of_lists_format
else:
    from feasibility_check_utils.jssp import validate_accord_format, validate_list_of_lists_format
# -----------------------------
# Load the Model and Tokenizer
# -----------------------------
#model on easy data untill 100 nodes included
if args.infer_fssp:
    model_dir = "fssp_models"
else:
    model_dir = "jssp_models"

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

if args.infer_fssp:
    eval_dataset = load_dataset("json", data_files='validation_data/fssp_val_data.json')

else:
    eval_dataset = load_dataset("json", data_files='validation_data/jssp_val_data.json')
    # eval_dataset = load_dataset("json", data_files='train_data/jssp_train_data/jssp_train_data1.json')


# eval_dataset = load_dataset("json", data_files="train_data/binpack_train_data/binpack_train_data.json")
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
                    "Num_J",
                    "Num_M",
                    "Declared_makespan",
                    "Actual_Makespan",
                    "Gap",
                    "Total_Sampling_Attempts",
                    "Num_Feasible_Solutions",
                    "Feasible_Sol_Makespans",
                    "Validation_Message",
                    "Path",
                    "Best_Solution",
                    "Avg_Time"  
                ]
            )
    real_feas_list = []
    gap_list = []
    
    print("start_idx", start_idx)
    for idx, example in tqdm(enumerate(dataset["train"])):
        total_feasible_solutions = 0  # Track the total number of feasible solutions found
        # Skip examples that have already been processed
        if idx < start_idx:
            continue
        # Example fields: "instruction", "input", "output", "num_items", "capacities", "path", etc.
        
        # if args.infer_fssp:
        problem_in_matrix_form = example["matrix"]
        n = example["num_jobs"]
        m = example["num_machines"]
        ms = example["makespan"]

    

        max_new_tokens = max_new_token_pred(
            example["num_jobs"], example["num_machines"]
        )
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
        print(n, m)

        # Create prompt without revealing the expected output.
        prompt = alpaca_prompt.format(example["instruction"], input, "")

        # print("*8**8**8**8**8**8**8**8**8*"*10)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to("cuda")

        feasible_solutions = []
        feasible_validation_messages = []
        num_feasible_solutions = 0
        total_sampling_attempts = 0
        infeasible_validation_message = ""
        total_time = 0.0  # Initialize total time for this instance
        feasible_makespans = []
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
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
            #for debugging
            # generated_texts = [real_solution]
            # End timing after generation.
            end_time = time.time()
            total_time += (end_time - start_time)

            # print("generated_texts", generated_texts)
            # Validate each generated solution.
            for k, gen_text in enumerate(generated_texts):
                if k == 0 and attempt == 0:
                    print(gen_text,flush=True)
                 
                is_feasible, message, declared_makespan = feasibility_checker(
                            example["input"], gen_text
                        )
                print("is_feasible", is_feasible, flush=True)
                if is_feasible:
                    feasible_solutions.append(gen_text)
                    feasible_makespans.append(declared_makespan)
                    feasible_validation_messages.append(message)
                    num_feasible_solutions += 1  # Increment feasible solutions counter
                    total_feasible_solutions += 1  # Increment total feasible solutions counter
                    gap = abs((declared_makespan - int(ms)) / ms)
                    print( f"GAP for problem {idx}: n={n}, m={m}, gap={gap}", flush=True)
                    
                    gap_list.append(gap)
                    print("SUM: ", sum(gap_list))

                    if gap <= 0.05:
                        found_threshold = True
                        break
                else:
                    infeasible_validation_message = message  # Keep the last infeasible message


            # Exit the outer generation loop if a sufficient solution is found.
            if found_threshold:
                 break

        
        # sys.exit()

        if feasible_solutions and feasible_makespans:
            integer_makespans = [int(x) for x in feasible_makespans]
            best_min_makespan = min(integer_makespans)
            min_index = integer_makespans.index(best_min_makespan)
            print("min_index", min_index)
            print("feasible_solutions", len(feasible_solutions))
            print("integer_makespans", len(integer_makespans))


            best_solution = feasible_solutions[min_index]
            validation_message = feasible_validation_messages[min_index]
            gap = abs((best_min_makespan - int(ms)) / ms)
            print(
                f"Best feasible solution found for problem {idx}: n={n}, m={m}, gap={gap}"
            )
        else:
            # If no feasible solution found after max_attempts
            best_solution = None
            validation_message = infeasible_validation_message
            declared_makespan = None  # Since no feasible makespan was found
            gap = None
        
        all_predictions.append(best_solution)
        # all_references.append([example["output"]])
        path = example.get("path", "")
        # Calculate the average time per attempt.
        avg_time = total_time / (max_attempts*num_return_sequences) if max_attempts > 0 else 0.0


        # Write results for the current instance.
        with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [   idx,
                    n,
                    m,
                    declared_makespan,
                    ms,
                    gap,
                    total_sampling_attempts,
                    num_feasible_solutions,
                    feasible_makespans,
                    validation_message,
                    path,
                    " ",
                    avg_time,
                    # best_solution

                ]
            )

    return all_predictions, all_references


# -----------------------------
# Main Execution
# -----------------------------
print("Starting evaluation loop...")
import os
model_base_name = os.path.basename(model_name) if model_name else "model"

if args.infer_fssp:
    output_dir = "val_results/fssp_val"
    if args.model_1B:
        output_dir = "val_results/fssp_val_1B"
else:
    output_dir = "val_results/jssp_val"
    if args.model_1B:
        output_dir = "val_results/jssp_val_1B"




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
