import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from data_prep import load_and_prepare_dataset
from model_setup import setup_model, setup_trainer
from inference import run_inference
from save_model import save_model

def main():
    # Configuration
    max_seq_length = 2048
    dtype = None  # None for auto detection
    load_in_4bit = True

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Setup model with LoRA
    model = setup_model(model)

    # Load and prepare dataset
    dataset = load_and_prepare_dataset(tokenizer)

    # Setup trainer
    trainer = setup_trainer(model, tokenizer, dataset, max_seq_length)

    # Train the model
    trainer_stats = trainer.train()

    # Print training stats
    print_training_stats(trainer_stats)

    # Run inference
    run_inference(model, tokenizer)

    # Save the model
    save_model(model, tokenizer)

def print_training_stats(trainer_stats):
    gpu_stats = torch.cuda.get_device_properties(0)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    
    print(f"Training time: {trainer_stats.metrics['train_runtime']} seconds")
    print(f"Training time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
    print(f"Peak reserved memory: {used_memory} GB")
    print(f"Peak reserved memory % of max memory: {round(used_memory/max_memory*100, 3)}%")

if __name__ == "__main__":
    main()
