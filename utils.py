

def predict_model(model, tokenizer, messages, configuration=None):
    ######################################
    ### STUB: INSERT THE CODE HERE###
    ######################################
    
    raise NotImplementedError("Build the Transformers package operations here based on the configurations given in the assignment")
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
