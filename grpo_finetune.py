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
    Returns True if the model_output's final numeric answer matches the correct answer. Note that some arguments in this function exist only to match the expected signature for TRL reward functions.
    Keep in mind that we use correct_answer name here instead of answer to avoid confusion with the model's output answer (which is the completions in this case).

    Args:
        prompts (dict): 8 identical prompts for each question.
        completions (str): 8 different completions from the model for each prompt. Their differences are used for relative comparisons.
        question (str): The math question being asked. Repeated 8 times to match the completions.
        correct_answer (str): The correct answer (0:A, 1:B, 2:C, or 3:D) for the question. Repeated 8 times to match the completions.
        args and kwargs: Additional arguments (not used here).

    Returns:
        True (1.0) if option A, B, C, or D matches at the last 20 characters of the output, False (0.0) otherwise.
        List of the above 0s or 1s are returned. By default, there are 8 completions to reward per math question. 
    """

    def extract_chosen_option(text: str):
        # Look for the last occurrence of A, B, C, or D in the last 20 characters.
        match = re.search(r'(?<![a-zA-Z])([ABCD])(?![a-zA-Z])', text[-20:])
        return match.group(1) if match else None
    # print(f"\n***** Prompts: {prompts}")
    # print(f"***** Completions: {completions}")
    # print("***** Question: ", question[0])
    # print("***** Correct_answer: ", correct_answer[0])

    ######################################
    ### STUB: INSERT THE CODE HERE ###
    ######################################

    raise NotImplementedError("Reward function for GRPO algorithm. It can handle multiple completions per question and rewards them comperatively. By default, it is n=8 repetitions.")


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

# print(ex_row['prompt'], "\n")

test_output = 8 * ["To solve this problem, we can use the formula for the number of edges in a complete graph with \\(n\\) vertices:\n\n\\[ E = \\frac{n(n - 1)}{2} \\]\n\nwhere \\(E\\) represents the total number of edges.\n\nGiven \\(n = 10\\), we can substitute \\(n = 10\\) into the formula:\n\n\\[ E = \\frac{10(10 - 1)}{2} \\]\n\\[ E = \\frac{10 \\times 9}{2} \\]\n\\[ E = 5 \\times 9 \\]\n\\[ E = 45 \\]\n\nTherefore, a complete graph with 10 vertices has 45 edges.\n\nMatched option: D<|im_end|>"]
rewards = reward_function([ex_row], test_output, [ex_row['question']], [ex_row['correct_answer']])
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
    ######################################
    ### STUB: INSERT THE CODE HERE ###
    ######################################
    
    raise NotImplementedError("Preprocess the dataset to create 'prompt' and 'correct_answer' fields.")


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
    with torch.no_grad():
        for ex in tqdm(examples, desc="Generating outputs", total=total, unit="ex"):
            question = ex.get("question", "")
            corr_answer = ex.get("correct_answer", "")
            choices = ex.get("choices", "")
            inputs = tokenizer(question, return_tensors="pt", truncation=True).to(device)

            gen = model_to_run.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )
            full_text = tokenizer.decode(gen[0], skip_special_tokens=True)
            # attempt to strip the prompt to get completion
            if full_text.startswith(question):
                completion = full_text[len(question):].strip()
            else:
                completion = full_text.strip()

            outputs.append({"question": question, "choices": choices, "correct_answer": corr_answer, "completion": completion})
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

training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct").to("cuda")

# Save base model outputs (before GRPO training)
base_out_dir = "./Qwen2-0.5B-GRPO"
base_out_name = "Qwen2-0.5B-base_OUTPUTS"
generate_and_save(model, eval_dataset, base_out_dir, base_out_name, max_new_tokens=2000)

# Initialize GRPO Trainer and start training
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_function,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()

# Save the model
trainer.save_model("Qwen2-0.5B-GRPO/Qwen2-0.5B-GRPO-Finetuned")

# Evaluate the model on the eval dataset
results = trainer.evaluate(eval_dataset=eval_dataset)
print("Evaluation results:", results)

# ============================================================
# Save finetuned model outputs (same 20 examples) AFTER GRPO training
# ============================================================
fine_out_dir = "./Qwen2-0.5B-GRPO"
fine_out_name = "Qwen2-0.5B-GRPO-finetuned_OUTPUTS"
# trainer.model is the trained model instance managed by TRL
trained_model = getattr(trainer, "model", model)
# use the same eval_subset (20 examples) instead of entire eval_dataset
# Note: You may change max_new_tokens to perform further experiments.
generate_and_save(trained_model, eval_dataset, fine_out_dir, fine_out_name, max_new_tokens=2000)
