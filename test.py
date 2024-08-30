import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer
from gptqmodel import GPTQModel

# ref model
ref_model_path = "NousResearch/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(ref_model_path)
model = AutoModelForCausalLM.from_pretrained(ref_model_path,
    torch_dtype=torch.float16, device_map='auto', load_in_8bit=False)
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


# 2-bit model in BitBLAS
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
