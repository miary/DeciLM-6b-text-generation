import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer


start = time.time()

model_id = 'Deci/DeciLM-6b'

model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map="auto",
                                             trust_remote_code=True
                                             )

tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token


def generate_text(prompt:str, max_new_tokens:int, temperature:float) -> str:
    model_inputs = tokenizer(prompt, return_tensors="pt").to("cpu") # gpu
    generated_ids = model.generate(**model_inputs,
                                   max_new_tokens=max_new_tokens,
                                   do_sample=True,
                                   temperature=temperature,
                                   # num_beams: widen the search space, allowing the model to explore multiple possibilities 
                                   # before settling on the final sequence. Higher value = higher computational overhead
                                   # num_beams= 5,
                                   # no_repeat_ngram_size: prevent repetition of n-grams. An n-gram is a contiguous sequence of 'n' items from a text 
                                   # significantly reduces repetitiveness in the outpu
                                   no_repeat_ngram_size=2,
                                   # Text generation stops once a logical endpoint is reached, even if the max_length hasn't been attained. 
                                   # Often leads to more concise and relevant outputs.
                                   early_stopping=True
                                   )
    decoded_generation = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return print(decoded_generation)

#prompt = """In this blog post, we're going to talk about the need to know and understand copyright laws when using social media"""
prompt = '''In this blog post, we're going to discuss the importance of knowing in depth the ramifications of 
the fourth amendment during a traffic stop. 
We will cite at least one related court case related to traffic stop and the fourth amendment'''

generate_text(prompt, 500, 0.25)


end = time.time()
print(f"\nExecution time: {(end-start):.02f} seconds\n")