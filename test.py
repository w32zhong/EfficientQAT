import time
from transformers import AutoTokenizer
from transformers import TextStreamer
from gptqmodel import GPTQModel

model_path = "ChenMnZ/Llama-2-7b-EfficientQAT-w2g64-BitBLAS"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = GPTQModel.from_quantized(model_path)
streamer = TextStreamer(tokenizer)

start = time.time()
output = model.generate(
    **tokenizer("Solar eclipse is ", return_tensors="pt").to(model.device),
    max_new_tokens=256, streamer=streamer, use_cache=True
)
end = time.time()

output_len = output.shape[-1]
delta_time = end - start
print(output_len, delta_time, output_len / delta_time)
