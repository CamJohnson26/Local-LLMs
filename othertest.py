import os
import torch
from transformers import pipeline, GPTNeoXTokenizerFast, GPTNeoXForCausalLM

model_name = 'EleutherAI/pythia-2.8b-deduped'

model = GPTNeoXForCausalLM.from_pretrained(model_name)
tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_name)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)

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

    print(generator(f"{context}{prompt}", do_sample=True, temperature=0.8, top_k=50, top_p=0.92, num_beams=5, max_new_tokens=100))
