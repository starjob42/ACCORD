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
from feasibility_check_utils.vrp_tsp import validate_accord_format, validate_list_of_lists_format

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
parser.add_argument('--infer_tsp', action='store_true',default=False, help='Weather to infer with TSP model or not')
parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'],  help='Data type (bfloat16 or float16)')
parser.add_argument('--load_in_4bit', action='store_true', default=True,  help='Use 4-bit quantization to reduce memory usage')
# Output directory
parser.add_argument('--output_dir', type=str, default=None, help='Output directory name')

args = parser.parse_args()

    

# -----------------------------
# Load the Model and Tokenizer
# -----------------------------
#model on easy data untill 100 nodes included
if args.output_accord:
    if args.model_1B:
        model_name = "finetuned_models/vrp_tsp_models/llama1B/accord/"
    else:
        model_name = "finetuned_models/vrp_tsp_models/llama8B/accord/"
elif args.output_list_of_lists:
    if args.model_1B:
        model_name = "finetuned_models/vrp_tsp_models/llama1B/list_of_lists/"
    else:
        model_name = "finetuned_models/vrp_tsp_models/llama8B/list_of_lists/"


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

def formatting_prompts_st_vrp_tsp(examples):
    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    instructions = examples["instruction"]
    inputs       = examples["input"]
    # inputs       = examples["paired_distances"]

    if args.use_distance:
        k = 0
        for i,j in zip(inputs, examples["paired_distances"]):
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
    return { "text" : texts, }


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_, output)
        texts.append(text)
    return {"text": texts}



# -----------------------------
# Load the Evaluation Dataset
# -----------------------------
if args.infer_tsp:
    eval_dataset = load_dataset("json", data_files='validation_data/tsp_val_data.json')
else:
    eval_dataset = load_dataset("json", data_files='validation_data/vrp_val_data.json')
# Apply the formatting function to the eval dataset
eval_dataset = eval_dataset.map(formatting_prompts_st_vrp_tsp, batched=True)

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
    # Check for already processed examples
    # Get the last processed index
    last_processed_index = get_last_processed_index(csv_filename)
    start_idx = last_processed_index + 1
    # If file does not exist, write header; else, open in append mode.
    write_header = not os.path.exists(csv_filename)

    # Open CSV file in append mode to continue from where we left off
    file_mode = "w" if write_header else "a"

    # Write CSV header.
    with open(csv_filename, mode=file_mode, newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            
            if write_header:
                writer.writerow(
                 [  "Index",
                    "Num_Vehicles",     
                    "Num_Cities",          
                    "Capacity",
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

    
    
    print("dataset[train])", len(dataset["train"]))
    print(f"Starting from example index: {start_idx}")

    

    for idx, example in tqdm(enumerate(dataset["train"])):
       
        # Skip examples that have already been processed
        if idx < start_idx:
            continue

        instruction = example.get("instruction")
        input       = example.get("input")

        num_cities = example.get("num_cities")
        num_vehicles = example.get("num_vehicles")
        capacity = example.get("capacity")

    
        print(f"Processing example {idx} ")
        print("num_cities ", num_cities)
        print("num_vehicles", num_vehicles)

        distance_matrix_str = example.get("paired_distances")


        if args.use_distance:
            input = "coordinates: " + input + "\n" + "distances : " + distance_matrix_str
       

        if args.output_accord:
            real_solution =  example.get("output_accord")
            feasibility_checker = validate_accord_format
        elif args.output_list_of_lists:
            real_solution =  example.get("output_list_of_lists")
            feasibility_checker = validate_list_of_lists_format
        
        # Create prompt without revealing the expected output.
        prompt = alpaca_prompt.format(instruction, input, "")

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to("cuda")

        feasible_solutions = []
        feasible_values = []
        feasible_validation_messages = []
        gap_list = []
        is_feasible_list = []

        num_feasible_solutions = 0
        total_sampling_attempts = 0
        infeasible_validation_message = ""
        total_time = 0.0  # Initialize total time for this instance
        
        found_threshold = False  # Flag to stop once gap is within the threshold.
        # Generate multiple attempts.
        for attempt in tqdm(range(max_attempts)):
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
            # generated_texts = ["Vehicle Route: (0): (93, 95) -> (2): (57, 116) + 41 -> (1): (20, 108) + 37 -> (4): (13, 114) + 9 -> (3): (18, 16) + 98 -> (0): (93, 95) + 108\nOverall Total Distance: 293"]

            #for debugging
            # generated_texts = [real_solution]



            # End timing after generation.
            end_time = time.time()
            total_time += (end_time - start_time)

            # print("generated_texts", generated_texts)
            # Validate each generated solution.
            for gen_text in generated_texts:

                is_feasible, message, gen_declared_value,gen_computed_value, _ = feasibility_checker(input, gen_text, capacity, num_vehicles, num_cities, distance_matrix_str)

                _, _, real_declared_value, optimal_value, _ = feasibility_checker(input, real_solution, capacity, num_vehicles, num_cities, distance_matrix_str
                    )
                # print("Real solution optimal_value:", optimal_value)
                # print("Real solution declared_value:", real_declared_value)
                print(gen_text)
                
                if is_feasible:
                    feasible_solutions.append(gen_text)
                    feasible_values.append(gen_computed_value)
                    feasible_validation_messages.append(message)
                    num_feasible_solutions += 1
                    
                    gap = abs((gen_computed_value-optimal_value) / optimal_value)
                    gap_list.append(gap)
                    is_feasible_list.append(is_feasible)
                    print("GAP: ", sum(gap_list))
                    print("average feasible_solutions",  sum(is_feasible_list)/len(is_feasible_list))

                    # Break if gap is within the acceptable threshold.
                    if gap <= 0.05:
                        found_threshold = True
                        break
                else:
                    infeasible_validation_message = message

            # Exit the outer generation loop if a sufficient solution is found.
            if found_threshold:
                 break 
            
        if feasible_solutions:
            # Choose the best solution (here using the highest declared value).
            best_index = np.argmin(gap_list)
            best_solution = feasible_solutions[best_index]
            best_declared_value = feasible_values[best_index]
            best_message = feasible_validation_messages[best_index]
            best_gap = gap_list[best_index]
            # Re-run validation to get the optimal value and gap.
            print("best_gap from list", best_gap)
            _, _, real_declared_value, optimal_value, _ = feasibility_checker(input, real_solution, capacity, num_vehicles, num_cities, distance_matrix_str
                    )
            _, _, gen_declared_value, best_found_value, _ = feasibility_checker(input, best_solution, capacity, num_vehicles, num_cities, distance_matrix_str                    )
            best_gap = abs((best_found_value - optimal_value)/optimal_value)
            
            print("num_vehicles",num_vehicles,"num_cities : ",num_cities,"GAP:", best_gap)
        else:
            best_solution = ""
            best_declared_value = None
            optimal_value = None
            best_gap = None
            best_message = infeasible_validation_message

        all_predictions.append(best_solution)
        path = example.get("path", "")
        # Calculate the average time per attempt.
        avg_time = total_time / (max_attempts*num_return_sequences) if max_attempts > 0 else 0.0

        
        # Write results for the current instance.
        with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [   idx,
                    num_vehicles,
                    num_cities,
                    capacity,
                    best_declared_value,
                    optimal_value,
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

    return all_predictions

# -----------------------------
# Main Execution
# -----------------------------
print("Starting evaluation loop...")
import os
model_base_name = os.path.basename(model_name) if model_name else "model"
output_dir = "val_results/vrp_val"


if args.model_1B:
    output_dir = "val_results/vrp_val_1B"

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
