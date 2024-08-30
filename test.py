from transformers import AutoTokenizer
from gptqmodel import GPTQModel

model_path = "ChenMnZ/Llama-2-7b-EfficientQAT-w2g64-BitBLAS"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = GPTQModel.from_quantized(model_path)

print(tokenizer.decode(
    model.generate(
        **tokenizer("Model quantization is", return_tensors="pt").to(model.device)
    )[0]
))
