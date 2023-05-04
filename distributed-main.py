# DEPRECATED: ALL THIS IS EASY JUST NO ONE TELLS YOU.
# Use main, and refer to my project 230225
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

organization = 'EleutherAi'
model_name = 'pythia-12b-deduped'

qualified_name = organization + '/' + model_name
checkpoint = qualified_name

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained('pythia-12b-deduped/main/models--EleutherAi--pythia-12b-deduped/snapshots/ac063574ab103137803591da0cc8d05854afa212',
                                             device_map='auto')

prompt = "What is the color of a carrot?\nA:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
generated_ids = model.generate(input_ids)
res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(res)