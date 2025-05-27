import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets import load_from_disk
from transformers import AutoTokenizer, get_scheduler
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import wandb
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dynamic_router.model import EnhancedTextClassifier

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an attention-based classifier for optimization problems"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        help="Base model to use",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument(
        "--data_path", type=str, default="./classifier_data/optimization_problems_dataset.json", help="Path to the dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, default="router_net/classifier_model_new", help="Directory to save the model"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="ACCORD_attention_classifier_vrp_tsp_joint",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=5, help="Number of steps between evaluations"
    )
    return parser.parse_args()

def tokenize_function(examples, tokenizer, max_length):
    # This returns a dict with 'input_ids' and 'attention_mask'
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    # Return the tokenized results to be added to the dataset
    return tokenized


def evaluate(model, dataloader, device, loss_fn, problem_types=None, step=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            # Only move tensor values to device or use only the needed columns
            filtered_batch = {
                k: v.to(device) if hasattr(v, 'to') else v
                for k, v in batch.items() 
                if k in ["input_ids", "attention_mask", "label"]
            }
            
            logits = model(
                input_ids=filtered_batch["input_ids"], 
                attention_mask=filtered_batch["attention_mask"]
            )
            loss = loss_fn(logits, filtered_batch["label"])
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = filtered_batch["label"].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    
    # Create and log confusion matrix if problem_types is provided
    metrics_dict = {
        "val_loss": avg_loss,
        "val_accuracy": accuracy,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1,
    }
    
    if problem_types is not None:
        cm = confusion_matrix(all_labels, all_preds)
        # Normalize for better visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create confusion matrix figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=problem_types.values(), 
                    yticklabels=problem_types.values())
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Log figure to wandb
        if step is not None:
            metrics_dict["confusion_matrix"] = wandb.Image(plt)
        plt.close()
        
        # Log per-class metrics
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        for i, class_name in problem_types.items():
            metrics_dict[f"precision_{class_name}"] = class_precision[i]
            metrics_dict[f"recall_{class_name}"] = class_recall[i]
            metrics_dict[f"f1_{class_name}"] = class_f1[i]
    
    # Log all metrics together using the provided step (if any)
    if step is not None:
        wandb.log(metrics_dict, step=step)
    
    return avg_loss, accuracy, precision, recall, f1


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define problem types mapping
    problem_types = {
        0: "binpacking",
        1: "flowshop",
        2: "jssp",
        3: "knapsack",
        4: "tsp-tsp",
        # 4: "vrp",
    }

    # Initialize W&B with fixed run name
    wandb.init(
        project=args.wandb_project,
        name="accord_classifier",
        config={
            **vars(args),
            "problem_types": problem_types,
            "model_type": "EfficientTextClassifier"
        },
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset from JSON files
    dataset = load_dataset("json", 
                        data_files={
                            "train": "router_net/classifier_data_vrp_tsp_same/train.jsonl",
                            "validation": "router_net/classifier_data_vrp_tsp_same/validation.jsonl",
                            "test": "router_net/classifier_data_vrp_tsp_same/test.jsonl"
                        })

    # Access the splits directly
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # Verify data loaded correctly
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Count class distribution for logging
    class_counts = {name: 0 for name in problem_types.values()}
    for split_name, split_data in [("train", train_dataset), ("validation", val_dataset), ("test", test_dataset)]:
        split_counts = {name: 0 for name in problem_types.values()}
        for item in split_data:
            class_name = problem_types[item["label"]]
            split_counts[class_name] += 1
            if split_name == "train":
                class_counts[class_name] += 1
        
        # Log class distribution for this split
        wandb.run.summary[f"{split_name}_distribution"] = split_counts
    
    # Create a table with class counts
    table = wandb.Table(columns=["Problem Type", "Count"])
    for class_name, count in class_counts.items():
        table.add_data(class_name, count)
    wandb.log({"class_distribution": table})

    # Tokenize datasets
    train_dataset = train_dataset.map(
        lambda examples: tokenize_function(
            examples, tokenizer=tokenizer, max_length=args.max_length
        ),
        batched=True,
    )
    val_dataset = val_dataset.map(
        lambda examples: tokenize_function(
            examples, tokenizer=tokenizer, max_length=args.max_length
        ),
        batched=True,
    )
    test_dataset = test_dataset.map(
        lambda examples: tokenize_function(
            examples, tokenizer=tokenizer, max_length=args.max_length
        ),
        batched=True,
    )

    # Set format for all datasets
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=SequentialSampler(val_dataset)
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Model, optimizer, scheduler, loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedTextClassifier(
            tokenizer_name=args.base_model,
            num_classes=len(problem_types),
            embedding_dim=768,
            hidden_dim=256,
            max_position=args.max_length  # Pass the max_length as max_position
        ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    num_steps = args.epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_steps,
    )
    loss_fn = nn.CrossEntropyLoss()

    # Log model architecture to wandb
    wandb.watch(model, log="all", log_freq=10)

    best_val_acc = 0.0
    global_step = 0
    progress_bar = tqdm(total=num_steps, desc="Training")

    # Initial validation
    print("Running initial validation...")
    val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
        model, val_loader, device, loss_fn, problem_types, step=global_step
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            # Move all tensors to the correct device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = loss_fn(logits, batch["label"])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            progress_bar.update(1)

            # Log training loss & lr every step (using global_step)
            wandb.log(
                {
                    "train_loss_step": loss.item(), 
                    "lr": optimizer.param_groups[0]["lr"]
                },
                step=global_step
            )

            # Validate periodically using consistent step numbering
            if global_step % args.eval_steps == 0:
                print(f"\nValidating at step {global_step}...")
                val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
                    model, val_loader, device, loss_fn, problem_types, step=global_step
                )
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.output_dir, "best_model.pt"),
                    )
                    wandb.run.summary["best_val_accuracy"] = best_val_acc
                    print(f"New best validation accuracy: {best_val_acc:.4f}")
                
                # Return to training mode
                model.train()
        
        # Log epoch-level metrics
        avg_epoch_loss = epoch_loss / len(train_loader)
        wandb.log(
            {
                "train_loss_epoch": avg_epoch_loss, 
                "epoch": epoch
            }, 
            step=global_step
        )
        
        print(f"Epoch {epoch}/{args.epochs} completed. Avg loss: {avg_epoch_loss:.4f}")
    
    progress_bar.close()

    # Final evaluation on test set
    print("Evaluating on test set...")
    test_metrics = {
        "test_loss": 0,
        "test_accuracy": 0,
        "test_precision": 0,
        "test_recall": 0,
        "test_f1": 0
    }
    
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
        model, test_loader, device, loss_fn, problem_types
    )
    
    test_metrics.update({
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_f1": test_f1
    })
    
    # Log test metrics at the final step
    wandb.log(test_metrics, step=global_step)
    
    print(
        f"Test loss: {test_loss:.4f} | "
        f"Accuracy: {test_acc:.4f} | "
        f"Precision: {test_prec:.4f} | "
        f"Recall: {test_rec:.4f} | "
        f"F1: {test_f1:.4f}"
    )
    
    # Add test metrics to run summary for easier access
    wandb.run.summary["test_accuracy"] = test_acc
    wandb.run.summary["test_f1"] = test_f1

    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "problem_types": problem_types,
            "final_metrics": test_metrics
        },
        os.path.join(args.output_dir, "final_model.pt"),
    )
    print("Training completed!")
    wandb.finish()

if __name__ == "__main__":
    main()