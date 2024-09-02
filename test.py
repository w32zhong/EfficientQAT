import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer
from gptqmodel import GPTQModel
from quantize.int_linear_real import load_quantized_model


# # ref model
# ref_model_path = "NousResearch/Llama-2-7b-hf"
# 
# tokenizer = AutoTokenizer.from_pretrained(ref_model_path)
# model = AutoModelForCausalLM.from_pretrained(ref_model_path,
#     torch_dtype=torch.bfloat16, device_map='auto', load_in_8bit=False)
# streamer = TextStreamer(tokenizer)
# 
# start = time.time()
# output = model.generate(
#     **tokenizer("Solar eclipse is ", return_tensors="pt").to(model.device),
#     max_new_tokens=256, streamer=streamer, use_cache=True
# )
# end = time.time()
# 
# output_len = output.shape[-1]
# delta_time = end - start
# print(output_len, delta_time, output_len / delta_time)
# del model
# torch.cuda.empty_cache()


# 2-bit model in triton
local_dir = '~/.cache/huggingface/hub/models--ChenMnZ--Llama-2-7b-EfficientQAT-w2g64/snapshots/db0ec4980cf28f9abbd24ad2e201dabf8bf37f64'
model, tokenizer = load_quantized_model(os.path.expanduser(local_dir), 2, 64)
streamer = TextStreamer(tokenizer)
model.to('cuda:1')
import myhook
collect = myhook.add(model, [
	'model.layers.6.mlp.gate_proj',
	'model.layers.6.mlp.down_proj',
	'model.layers.8.mlp.gate_proj',
	'model.layers.8.mlp.down_proj',
	'model.layers.9.self_attn.q_proj',
	'model.layers.9.self_attn.o_proj',
])

start = time.time()
output = model.generate(
    **tokenizer("Solar eclipse is ", return_tensors="pt").to(model.device),
    max_new_tokens=256, streamer=streamer, use_cache=True
)
end = time.time()
output_len = output.shape[-1]
delta_time = end - start
print(output_len, delta_time, output_len / delta_time)
myhook.save(collect)
del model
torch.cuda.empty_cache()


# # 2-bit model in BitBLAS
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# model_path = "ChenMnZ/Llama-2-7b-EfficientQAT-w2g64-BitBLAS"
# 
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
# model = GPTQModel.from_quantized(model_path)
# model.to('cuda')
# streamer = TextStreamer(tokenizer)
# 
# start = time.time()
# output = model.generate(
#     **tokenizer("Solar eclipse is ", return_tensors="pt").to(model.device),
#     max_new_tokens=256, streamer=streamer, use_cache=True
# )
# end = time.time()
# 
# output_len = output.shape[-1]
# delta_time = end - start
# print(output_len, delta_time, output_len / delta_time)
# del model
# torch.cuda.empty_cache()
