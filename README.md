# LlamaChinese-Llama-3.1-Fine-tuning-for-Modern-to-Classical-Chinese-Translation
This project fine-tunes the Llama-3.1 8B model to translate modern Chinese to classical Chinese. 

## Features

- Uses Llama-3.1 8B model with 4-bit quantization
- Implements LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Supports both local and online model saving
- Includes inference capabilities for testing the model

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (for faster training and inference)

## Project Structure

- `main.py`: The main script that orchestrates the entire process
- `data_prep.py`: Handles dataset loading and preparation
- `model_setup.py`: Sets up the model and training configuration
- `inference.py`: Manages the inference process
- `save_model.py`: Handles model saving (local and online)
- `convertion.py`: Handles convert the dataset into json file

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
The raw data is from https://github.com/NiuTrans/Classical-Modern
The model is based on Llama 3.1 8B model
