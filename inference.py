from transformers import TextStreamer

def run_inference(model, tokenizer):
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    FastLanguageModel.for_inference(model)
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            "请把现代汉语翻译成古文",
            "乐极生悲",
            "",
        )
    ], return_tensors="pt").to("cuda")

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)
