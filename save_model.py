def save_model(model, tokenizer):
    # Local saving
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")

    # Online saving (uncomment and replace with your own token)
    model.push_to_hub("your-username/model-name", token="your-token")
    tokenizer.push_to_hub("your-username/model-name", token="your-token")

    # GGUF saving (uncomment and adjust as needed)
    model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
    model.push_to_hub_gguf("your-username/model-name", tokenizer, quantization_method="q4_k_m", token="your-token")
