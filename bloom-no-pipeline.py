# https://github.com/EleutherAI/pythia
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import AutoConfig, AutoModel, GPTNeoXTokenizerFast, GPTNeoXForCausalLM
# from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPTJForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig

print('Cuda devices available: ')
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))

organization = 'bigscience'
model_name = 'bloom-7b1'

qualified_name = organization + '/' + model_name

# device_map = {'gpt_neox.embed_in': 0, 'gpt_neox.layers.0': 0, 'gpt_neox.layers.1': 0, 'gpt_neox.layers.2': 0, 'gpt_neox.layers.3': 0,
#               'gpt_neox.layers.4': 1, 'gpt_neox.layers.5': 1, 'gpt_neox.layers.6': 1, 'gpt_neox.layers.7': 1, 'gpt_neox.layers.8': 1,
#               'gpt_neox.layers.9': 1, 'gpt_neox.layers.10': 1, 'gpt_neox.layers.11': 1, 'gpt_neox.layers.12': 1, 'gpt_neox.layers.13': 1,
#               'gpt_neox.layers.14': 1, 'gpt_neox.layers.15': 1, 'gpt_neox.layers.16': 1, 'gpt_neox.layers.17': 1, 'gpt_neox.layers.18': 1,
#               'gpt_neox.layers.19': 2, 'gpt_neox.layers.20': 2, 'gpt_neox.layers.21': 2, 'gpt_neox.layers.22': 2, 'gpt_neox.layers.23': 2,
#               'gpt_neox.layers.24': 2, 'gpt_neox.layers.25': 2, 'gpt_neox.layers.26': 2, 'gpt_neox.layers.27': 2, 'gpt_neox.layers.28': 2,
#               'gpt_neox.layers.29': 2, 'gpt_neox.layers.30': 2, 'gpt_neox.layers.31': 2, 'gpt_neox.layers.32': 2,
#               'gpt_neox.layers.33': 2, 'gpt_neox.layers.34': 2, 'gpt_neox.layers.35': 2, 'gpt_neox.final_layer_norm': 2,
#               'embed_out': 2}

checkpoint = "bloom-7b1/pytorch_model.bin.index.json"

config = AutoConfig.from_pretrained(qualified_name)

with init_empty_weights():
    model = AutoModel.from_config(config)

model = load_checkpoint_and_dispatch(
    model, checkpoint, device_map='balanced_low_0'
)
print(model.hf_device_map)

tokenizer = GPTNeoXTokenizerFast.from_pretrained(
    qualified_name,
    torch_dtype=torch.float16,
)

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
    input_ids = tokenizer(f"{context}{prompt}", return_tensors="pt").input_ids.to('cuda') #.to(device)

    tokens = model.generate(input_ids, do_sample=True, temperature=0.8, top_k=50, top_p=0.92, max_new_tokens=100)
    gen_text = tokenizer.batch_decode(tokens)[0]
    print(gen_text)
