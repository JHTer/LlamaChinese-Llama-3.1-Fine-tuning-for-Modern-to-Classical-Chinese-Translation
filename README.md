# LlamaChinese Llama 3.1 Fine-tuning for Modern to Classical Chinese Translation
This project fine-tunes the Llama-3.1 8B model to translate modern Chinese to classical Chinese. 

## Demonstration
#### Input:
```
轻轻的我走了，正如我轻轻的来；我轻轻的招手，作别西天的云彩。
那河畔的金柳，是夕阳中的新娘；波光里的艳影，在我的心头荡漾。
软泥上的青荇，油油的在水底招摇；在康河的柔波里，我甘心做一条水草！
那榆荫下的一潭，不是清泉，是天上虹；揉碎在浮藻间，沉淀着彩虹似的梦。
寻梦？撑一支长篙，向青草更青处漫溯；满载一船星辉，在星辉斑斓里放歌。
但我不能放歌，悄悄是别离的笙箫；夏虫也为我沉默，沉默是今晚的康桥！
悄悄的我走了，正如我悄悄的来；我挥一挥衣袖，不带走一片云彩。
```
#### Output:
```
轻轻步出，何其轻轻入也; 招手轻轻，别西天之浮云。
河畔有金柳，作夕阳之妇; 波上有艳色，荡我之心。
青荇浮于软泥上，油油若游水中; 在康河柔波之中，为水草者乐！
榆荫下一潭，不清泉，实虹也; 揉碎藻中，淀彩虹之梦也。
欲寻梦乎？扁舟轻泛青草更加青之水; 载满星辉，歌于星辉中。
然而吾不能歌，而悚悚为别笙也; 蟲蟲亦寗寗，寂寞者今夜康橋也!
悚悚我去，何其悚悚来也; 挥衫不遗片云。
```

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
