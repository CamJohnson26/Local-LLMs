import os
import torch
from transformers import pipeline, GPTNeoXTokenizerFast, GPTNeoXForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-3b")

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-3b", device_map='balanced_low_0')
print(model.hf_device_map)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

generator.tokenizer.pad_token_id = model.config.eos_token_id

context = f"""
###
Document:
<begin>
name,age,reputation
Stephen King,unknown,good
<end>
Question: What is this document
Answer: A short list of authors with their age and reputation.
###
"""

end_context = """
###
"""

while True:
    document = []
    while True:

        line = input('Enter Document') if len(document) == 0 else input()
        if line == "done":
            break
        document.append(line)
    document = "\n".join(document)
    question = input('Enter Question')

    prompt = f"""
    Document:
    <begin>
    {document}
    <end>
    Question: {question}
    Answer:"""

    print(generator(f"{context}{prompt}", do_sample=True, temperature=0.8, top_k=50, top_p=0.92, max_new_tokens=100))
