import torch # Make sure torch is imported

def predict_model(model, tokenizer, messages, configuration=None):
    ######################################
    ### STUB: INSERT THE CODE HERE###
    ######################################
    
    # Get the configuration settings. The instructions mention
    # temperature (0.1 or 0.7) and max_token_limit (2000 or 4000) [cite: 119, 120]
    if configuration is None:
        # Set defaults as per the function's docstring
        temp = 0.1
        max_toks = 2000
    else:
        temp = configuration.get("temperature", 0.1)
        max_toks = configuration.get("max_token_limit", 2000)

    # 1. Format the 'messages' list into a prompt string
    # The tokenizer's 'apply_chat_template' does this for us.
    prompt_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True, # This adds the "assistant" token to signal a reply
        return_tensors="pt"
    )
    
    # Move the formatted inputs to the same device as the model (e.g., the GPU)
    prompt_inputs = prompt_inputs.to(model.device)
    
    # 2. Call the model's 'generate' function
    # We must pass the configuration values here [cite: 110]
    # We also set do_sample=True, which is required for 'temperature' to have an effect
    outputs = model.generate(
        prompt_inputs,
        max_new_tokens=max_toks,
        temperature=temp,
        do_sample=True if temp > 0 else False
    )
    
    # 3. Decode the result
    # The 'outputs' tensor contains the *entire* conversation (prompt + response).
    # We only want the *new* tokens (the response part).
    
    # Find the length of the input prompt
    input_length = prompt_inputs.shape[1]
    
    # Slice the output tensor to get only the new tokens
    new_tokens = outputs[0][input_length:]
    
    # Decode the new tokens into a string
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return response.strip()

    """
    This function, `predict_model`, is designed to interact with QWEN models to generate predictions
    based on a conversation history. 

    Args:
        model: The pre-trained language model to be used for generating responses.
        tokenizer: the tokenizer corresponding to the model.
        messages: A list of dictionaries representing the conversation history,
                  where each dictionary has a "role" (e.g., "system", "user", or "assistant") 
                  and "content" (the message text).
        configuration: initially, the model used should be max_token_limit of 2000, with temperature of 0.1
        The assessment would mainly be assessed the correctness of the implementation, rather than the performance

    Returns:
        The model's response as a string.
    """


def model_evaluation(model_type, model, tokenizer, system_content, question, formatted_options, configuration=None):
    if model_type == "qwen2" or model_type == "qwen3":
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
        ]
        model_result = predict_model(model, tokenizer, messages, configuration)
    else: 
        raise ValueError(f"Unknown model_type: {model_type}")

    #  print(f"Model result: {model_result}")
    return model_result
