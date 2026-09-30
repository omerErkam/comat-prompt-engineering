import json
import os
import argparse
import torch
from dotenv import load_dotenv
from mmlu_redux import process_mmlu_redux_questions
import pandas as pd
from CoMAT_Instruction import INSTRUCTION
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    DATASET_CHOICES = [
        "mmlu-redux-college_mathematics"
    ]
    parser = argparse.ArgumentParser(description="Process MMLU questions")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="mmlu-redux-college_mathematics", help="Choose the dataset")
    parser.add_argument("--method", choices=["comat"], default="comat", help="Choose the method")
    parser.add_argument("--model", choices=["qwen3", "qwen2"], default="qwen2", help="Choose the model")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature setting for the model")
    parser.add_argument("--max_token_limit", type=int, default=2000, help="Max token limit for the model during generation")
    args = parser.parse_args()

    output_dir = f"final_results/{args.dataset}/{args.method}/{args.model}"
    output_file_path = f"{output_dir}/{args.method}_{args.model}.json"
    log_file_path = f"{output_dir}/{args.method}_{args.model}_log.txt"

    ensure_dir(output_file_path)

    with open(output_file_path, 'w') as f:
        json.dump([], f)
    print(f"Created output file: {output_file_path}")

    with open(log_file_path, 'w') as f:
        f.write(f"Start evaluating the {args.dataset} dataset with {args.method} method using {args.model} model. "
                f"(temperature={args.temperature}, max_token_limit={args.max_token_limit})\n")
    print(f"Created log file: {log_file_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == "qwen3":
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B").to(device)
    elif args.model == "qwen2":
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct").to(device)
    else:
        raise ValueError("Please choose a correct model")
    
    csv_path = "mmlu-redux-college_mathematics_dataset.csv"
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV file not found: {csv_path}")
    dataset = pd.read_csv(csv_path)

    configuration = {
        "temperature": args.temperature,
        "max_token_limit": args.max_token_limit
    }

    results, accuracy = process_mmlu_redux_questions(
        dataset=dataset,
        output_file_path=output_file_path,
        formulation_prompt=INSTRUCTION,
        model_type=args.model,
        model=model,
        tokenizer=tokenizer,
        configuration=configuration
    )

    print(results)
    print(f"Final results saved to {output_file_path}")
    print(f"Final Accuracy: {accuracy:.2%}")

    with open(log_file_path, 'a') as f:
        f.write(f"Final Accuracy: {accuracy:.2%}\n")

    print(f"Log file updated: {log_file_path}")


if __name__ == "__main__":
    main()
