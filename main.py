import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer

checkpoint = "Deci/DeciLM-6b"
device = "cpu" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

prompt = """Dear recruiter, I write this letter of recommendation for my toddler
son for his application to the Hogwarts School of Monster Trucks and Classic Cars.
He has over 100 monster trucks and this is beyond an obsession
"""

inputs = tokenizer.encode("In a shocking finding, scientists discovered a herd of unicorns living in", return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_p=0.95)
print(tokenizer.decode(outputs[0]))