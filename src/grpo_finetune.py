import re
import pandas as pd
from datasets import Dataset
from CoMAT_Instruction import INSTRUCTION
from trl import GRPOConfig, GRPOTrainer
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def reward_function(prompts: list, completions: list, question : list, correct_answer : list, *args, **kwargs) -> list:
    """
    Returns True if the model_output's final numeric answer matches the correct answer. 
    
    Args:
        prompts (list): List of 8 identical prompts for each question.
        completions (list): List of 8 different completions from the model.
        question (list): List of 8 identical questions.
        correct_answer (list): List of 8 identical correct answers.
    
    Returns:
        list: List of rewards (1.0 for correct, 0.0 for incorrect).
    """

    def extract_chosen_option(text: str):
        # Look for the last occurrence of A, B, C, or D in the last 20 characters.
        match = re.search(r'(?<![a-zA-Z])([ABCD])(?![a-zA-Z])', text[-20:])
        return match.group(1) if match else None

    rewards = []
    
    # Iterate through each completion and corresponding correct answer
    for i, completion in enumerate(completions):
        # Extract the model's predicted letter
        prediction = extract_chosen_option(completion)
        
        # Get the ground truth for this specific example
        # correct_answer is a list corresponding to the batch
        truth = correct_answer[i]
        
        # Compare and assign reward
        if prediction == truth:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards


# ------------------------------------------------ EXAMPLE SCENARIO for reward function ------------------------------------------------
print("~~~~~~ Testing reward function with a toy example...\n")
# Test the reward function on a toy example
ex_row = {
    'question': 'A discrete graph is complete if there is an edge connecting any pair of vertices. How many edges does a complete graph with 10 vertices have?',
    'choices': "['A. 10', 'B. 20', 'C. 25', 'D. 45']",
    'answer': 'D'
}

# Extract and format the options
ex_row['choices'] = eval(ex_row['choices'])  # Convert string representation of list to actual list

formatted_options = "\n".join(
        [f"{opt}" for i, opt in enumerate(ex_row['choices'])]
)

comat_instruction = INSTRUCTION

ex_row['prompt'] = f"{comat_instruction}\n\n-----------\n\nQuestion: {ex_row['question']}\n\nOptions:\n{formatted_options}"
ex_row['correct_answer'] = ex_row['answer']

# Create a batch of 8 identical examples to simulate GRPO batching
# This matches the professor's requirement for 8-fold repetition
test_prompts = [ex_row['prompt']] * 8
test_output = 8 * ["To solve this problem, we can use the formula for the number of edges in a complete graph with \\(n\\) vertices:\n\n\\[ E = \\frac{n(n - 1)}{2} \\]\n\nwhere \\(E\\) represents the total number of edges.\n\nGiven \\(n = 10\\), we can substitute \\(n = 10\\) into the formula:\n\n\\[ E = \\frac{10(10 - 1)}{2} \\]\n\\[ E = \\frac{10 \\times 9}{2} \\]\n\\[ E = 5 \\times 9 \\]\n\\[ E = 45 \\]\n\nTherefore, a complete graph with 10 vertices has 45 edges.\n\nMatched option: D<|im_end|>"]
test_questions = [ex_row['question']] * 8
test_answers = [ex_row['correct_answer']] * 8

rewards = reward_function(test_prompts, test_output, test_questions, test_answers)
print(f"~~~~~~ Rewards for toy example output: {rewards} \n")

# ----------------------------------------------------------------------------------------------

# Load dataset from CSV (mmlu-redux-college_mathematics_dataset.csv)
csv_path = "mmlu-redux-college_mathematics_dataset.csv"
df = pd.read_csv(csv_path)

# Transform DataFrame to Hugging Face Dataset. Only keep these columns: ['question', 'choices', 'answer'].
dataset = Dataset.from_pandas(df[['question', 'choices', 'answer']])
print(dataset, "\n")

# Preprocess the dataset to create 'prompt' and 'correct_answer' fields as seen in the above example.
def preprocess_function(examples):
    prompts = []
    correct_answers = []
    
    # Iterate over the batch of examples
    for q, c, a in zip(examples['question'], examples['choices'], examples['answer']):
        # The choices are loaded as a string representation of a list, so we eval them
        options_list = eval(c)
        
        # Format the options A. ... B. ...
        formatted_options = "\n".join(
            [f"{opt}" for i, opt in enumerate(options_list)]
        )
        
        # Construct the full prompt string matching the CoMAT format
        prompt = f"{INSTRUCTION}\n\n-----------\n\nQuestion: {q}\n\nOptions:\n{formatted_options}"
        
        prompts.append(prompt)
        correct_answers.append(a)
        
    return {"prompt": prompts, "correct_answer": correct_answers}


dataset = dataset.map(preprocess_function, batched=True, remove_columns=['answer'])
print(dataset, "\n")


# ==============================================
# helper function: generate and save outputs
# ==============================================
def generate_and_save(model_to_run, examples, out_dir, out_name, max_new_tokens=None):
    """
    examples: list of dicts with keys "prompt" and "answer"
    Writes JSON list of {prompt, answer, completion} to out_dir/out_name.json
    """
    os.makedirs(out_dir, exist_ok=True)
    outputs = []
    model_to_run.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total = len(examples) if hasattr(examples, "__len__") else None
    
    # Ensure tokenizer is available
    if 'tokenizer' not in globals():
        raise ValueError("Tokenizer not found in globals.")

    with torch.no_grad():
        for ex in tqdm(examples, desc="Generating outputs", total=total, unit="ex"):
            # We need to extract the prompt text from the dataset example
            prompt_text = ex["prompt"]
            corr_answer = ex["correct_answer"]
            
            # Prepare inputs for Qwen model
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)

            gen = model_to_run.generate(
                model_inputs.input_ids,
                max_new_tokens=max_new_tokens,
                attention_mask=model_inputs.attention_mask
            )
            
            # Decode response
            gen_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, gen)
            ]
            completion = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

            outputs.append({"question": ex.get("question", ""), "correct_answer": corr_answer, "completion": completion})
            
    out_path = os.path.join(out_dir, f"{out_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(outputs)} outputs to {out_path}")
    return outputs


# Split dataset into train and eval (80-20 split) with seed=42
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']
print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)} \n")

# GRPO Config - using defaults as requested, just setting output dir
training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct").to("cuda")

# Save base model outputs (before GRPO training)
base_out_dir = "./Qwen2-0.5B-GRPO"
base_out_name = "Qwen2-0.5B-base_OUTPUTS"
# Using a subset of eval_dataset for speed if desired, but instructions imply the whole set
generate_and_save(model, eval_dataset, base_out_dir, base_out_name, max_new_tokens=2000)

# Initialize GRPO Trainer and start training
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_function,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer 
)
trainer.train()

# Save the model
trainer.save_model("Qwen2-0.5B-GRPO/Qwen2-0.5B-GRPO-Finetuned")

# Evaluate the model on the eval dataset (optional check)
# results = trainer.evaluate(eval_dataset=eval_dataset)
# print("Evaluation results:", results)

# ============================================================
# Save finetuned model outputs (same 20 examples) AFTER GRPO training
# ============================================================
fine_out_dir = "./Qwen2-0.5B-GRPO"
fine_out_name = "Qwen2-0.5B-GRPO-finetuned_OUTPUTS"
# trainer.model is the trained model instance managed by TRL
trained_model = getattr(trainer, "model", model)

generate_and_save(trained_model, eval_dataset, fine_out_dir, fine_out_name, max_new_tokens=2000)
