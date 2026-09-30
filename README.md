# CoMAT: Enhancing LLM Mathematical Reasoning & Alignment

## 📌 Project Overview
Large Language Models (LLMs) frequently struggle with strict mathematical reasoning and multi-step logic. This project explores and implements the **Chain of Mathematically Annotated Thought (CoMAT)** prompt engineering technique to improve LLM problem-solving capabilities. Beyond prompting, this project programmatically evaluates the marginal contribution of individual reasoning steps using **Shapley Value Analysis** and fine-tunes a smaller model using **Generalized Reinforcement Policy Optimization (GRPO)** to align its outputs with mathematically sound reasoning paths.

## 🛠 Tech Stack
*   **Languages:** Python (3.12)
*   **Deep Learning & NLP:** PyTorch, Hugging Face Transformers, `trl` (Transformer Reinforcement Learning)
*   **Models Evaluated:** Qwen2 (0.5B, 1.5B Instruct), Qwen3 (0.6B, 1.7B)
*   **Data & Evaluation:** `datasets` (MMLU-Redux), `pandas`, `numpy`, `sympy`, Regex parsing
*   **Environment:** CUDA-enabled GPU acceleration

## 🔬 Methodology
1.  **CoMAT Prompt Engineering Pipeline:**
    *   **Identification & Definition:** Extracting variables and constants from natural language problems.
    *   **Structural Logic Translation:** Translating relationships into formal algebraic equations.
    *   **Explicit Factual Representation:** Assigning known values to variables.
    *   **Question Formalization:** Defining the ultimate mathematical goal before execution.
2.  **Shapley Value Step Evaluation:** 
    *   Constructed a custom algorithm to calculate the Shapley value of each CoMAT step.
    *   Evaluated all permutations of prompt instructions on a held-out validation set to quantify the exact marginal contribution of each reasoning step to the model's final accuracy.
3.  **RLHF Finetuning (GRPO):**
    *   Applied the GRPO algorithm to optimize the policy network by generating multiple completions per prompt.
    *   Utilized a custom reward function based on formatting (+0.2), reasoning (+0.3), and final answer extraction (+0.5) to assign relative preference scores, stabilizing training without requiring absolute numeric rewards.

## 📊 Results & Evaluation

| Model | Token Limit | Temperature | Accuracy |
| :--- | :--- | :--- | :--- |
| **QWEN2** | 2000 | 0.1 | 34.34% |
| **QWEN2** | 2000 | 0.7 | 33.33% |
| **QWEN3** | 2000 | 0.1 | 42.41% |
| **QWEN3** | 4000 | 0.1 | 62.63% |

*   **Model Scaling & Context Limits:** QWEN3 achieved a peak accuracy of **62.63%** when the maximum token limit was expanded to 4000, significantly outperforming QWEN2's baseline of ~34%.
*   **Temperature Sensitivity:** Increasing inference temperature from 0.1 to 0.7 degraded performance, confirming that higher entropy introduces logic errors in strict algorithmic tasks.
*   **Shapley Value Insights:** "Question Formalization" (Step 4) yielded the highest positive contribution (-0.0192 relative weight) by acting as a critical focal point before execution.
*   **Ablation Study:** Removing both setup steps (Steps 1 & 2) resulted in a 4.48% *higher* accuracy than removing only Step 1, indicating that models perform better skipping the setup entirely rather than reasoning with a fractured or incomplete logic structure.

## 📂 Repository Structure

```text
├── data/
│   ├── raw/                        # MMLU-redux dataset files
│   └── evaluation_with_steps.csv   # Toy dataset for Shapley permutations
├── notebooks/
│   └── comat_exploration.ipynb     # Main experimental notebook
├── src/
│   ├── main.py                     # Entry point for CoMAT evaluation
│   ├── utils.py                    # Helper functions for model loading and QWEN predictions
│   ├── mmlu_redux.py               # Dataset processing and regex evaluation logic
│   ├── shapley_value_evaluation.py # Algorithm for calculating Shapley values
│   ├── comat_instruction.py        # Prompt templates and formatting logic
│   └── grpo_finetune.py            # Reinforcement learning training script
├── README.md
└── requirements.txt
