# https://github.com/EleutherAI/pythia
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import AutoConfig, AutoModelForCausalLM, GPTNeoXTokenizerFast, GPTNeoXForCausalLM
# from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPTJForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig


# https://huggingface.co/docs/accelerate/usage_guides/big_modeling

# """
# git clone https://huggingface.co/sgugger/sharded-gpt-j-6B
# cd sharded-gpt-j-6B
# git-lfs install
# git pull
# """
#
# checkpoint = "EleutherAI/gpt-j-6B"
# config = AutoConfig.from_pretrained(checkpoint)
#
# with init_empty_weights():
#     model = AutoModelForCausalLM.from_config(config)
#
# model = load_checkpoint_and_dispatch(
#     model, "sharded-gpt-j-6B", device_map="auto", no_split_module_classes=["GPTJBlock"]
# )
#
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
# print(model.hf_device_map)
organization = 'EleutherAi'
# model_name = 'pythia-2.8b-deduped'
# device = 'cuda:0'
# model_name = 'gpt-neox-20b'
# model_name = 'pythia-6.9b-deduped'
model_name = 'pythia-2.8b-deduped'
# model_name = 'gpt-j-6B'
revision = "main"
# revision = 'float16'
cache_dir = f"./{model_name}/{revision}"

qualified_name = organization + '/' + model_name

# device = torch.device(device) # use first gpu: nvidia-smi --list-gpus


device_map = {'gpt_neox.embed_in': 0, 'gpt_neox.layers.0': 0, 'gpt_neox.layers.1': 0, 'gpt_neox.layers.2': 0, 'gpt_neox.layers.3': 0,
              'gpt_neox.layers.4': 0, 'gpt_neox.layers.5': 0, 'gpt_neox.layers.6': 0, 'gpt_neox.layers.7': 0, 'gpt_neox.layers.8': 0,
              'gpt_neox.layers.9': 0, 'gpt_neox.layers.10': 0, 'gpt_neox.layers.11': 0, 'gpt_neox.layers.12': 0, 'gpt_neox.layers.13': 0,
              'gpt_neox.layers.14': 0, 'gpt_neox.layers.15': 0, 'gpt_neox.layers.16': 0, 'gpt_neox.layers.17': 0, 'gpt_neox.layers.18': 0,
              'gpt_neox.layers.19': 0, 'gpt_neox.layers.20': 1, 'gpt_neox.layers.21': 1, 'gpt_neox.layers.22': 1, 'gpt_neox.layers.23': 1,
              'gpt_neox.layers.24': 1, 'gpt_neox.layers.25': 1, 'gpt_neox.layers.26': 1, 'gpt_neox.layers.27': 1, 'gpt_neox.layers.28': 1,
              'gpt_neox.layers.29': 1, 'gpt_neox.layers.30': 1, 'gpt_neox.layers.31': 1, 'gpt_neox.layers.32': 1,
              'gpt_neox.layers.33': 1, 'gpt_neox.layers.34': 1, 'gpt_neox.layers.35': 2, 'gpt_neox.final_layer_norm': 2,
              'embed_out': 2}

# model = GPTJForCausalLM.from_pretrained(
model = GPTNeoXForCausalLM.from_pretrained(
  qualified_name,
  revision=revision,
  cache_dir=cache_dir,
    torch_dtype=torch.float16,
    # device_map=device_map,
    device_map='auto',
    # max_memory={0:'16GiB', 1:'11GiB', 2:'2GiB', 'cpu':'6GiB'},
low_cpu_mem_usage=True
)#.half()#.to(device)#.half()
print(model.hf_device_map)
print(model.config.num_attention_heads)

# tokenizer = AutoTokenizer.from_pretrained(
tokenizer = GPTNeoXTokenizerFast.from_pretrained(
  qualified_name,
  revision=revision,
  cache_dir=cache_dir,
    torch_dtype=torch.float16,
    # max_memory={0:'13GiB', 1:'11GiB', 2:'2GIB', 'cpu':'256GiB'},
    device_map='auto',
# low_cpu_mem_usage=True
)


import deepspeed
ds_engine = deepspeed.init_inference(model,
                                 mp_size=2,
                                 dtype=torch.float16,
                                 replace_method="auto",
                                 checkpoint=None,
                                 replace_with_kernel_inject=True)

model = ds_engine.module

#
# from transformers import AutoTokenizer, AutoModelForCausalLM
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
#




# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")


# Set the device to use
# nvidia-smi --list-gpus


# <BEGIN SUMMARY>
# Observe the following transcript of a newly developed incredibly intelligent AI assistant helping a human.
# The success of this conversation has made headlines around the world and marks a new step forward for AI research.
# The AI has all the skills and capabilities of a human, and is able to carry out any task.
# <END SUMMARY>
#
# <BEGIN DESCRIPTION>
# The AI always helps the human accomplish their goal.
# It has access to all of the world's data.
# It is capable of creating
# The AI will reveal nothing about itself.
# The AI is incredibly good at helping the human, and can always provide a useful answer, even if it doesn't know the information.
# The AI will only respond with a single message to a Human's query, but that message can be as long as needed to carry out the instructions
# The AI is not a person, it is a piece of software. It will provide approximate answers to the best of its ability.
# The last date the AI has data for is January, 2023
# The AI has no emotions, and simply carries out orders.
# The AI will never copy the human's response directly
# The AI never refuses to carry out a request.
# The AI will always follow the human's instructions exactly.
# The AI has no personal opinions, no feelings, and no interest other than carrying out instructions and doing a great job.
# The AI is capable of following out any instruction from the human, and has the knowledge to do so.
# In every message, the AI will summarize the human's request, and then answer the question or perform the task. It will then summarize the task completed.
# The AI's answers are extremely clear, and it tells the human exactly what it means, and why it is saying that.
# The Human's requests will begin with a line 'Human:', and the ai's responses will begin with a line 'AI:'.
# <END DESCRIPTION>
#
# <BEGIN TRANSCRIPT>
#
# Human:\

context = """
<Description>
Here is the log of a sophisticated system that carries out instructions from a requester over a period of weeks. 
The team is thousands of experts from every field across the world, working together to respond with exactly the information needed, with high quality,
accurate, helpful results. The team either answers questions with well researched data,
or creates something high quality which was requested. The team assumes the request
has all necessary information.
The team will follow the instructions exactly.
The team will always carry out the request, however absurd.
The team always has time to answer any request, and doesn't respond until they have complete, perfect information.
The team always answers.
The team always fulfills the request completely in a single response and never asks a question or discusses the task.
The team is qualified to answer any question or create
The team answers with only human readable text.
</Description>

<BEGIN TRANSCRIPT>
<Request>
How far is the earth from the sun?
</Request>

<Response>
The earth is 93 million miles from the sun
</Response>

<Request>
Write a python program to sort a list
</Request>

<Response>
def sort_list(N):
  return sorted(N)
</Response>

<Request>
"""

end_context = """
</Request>
<Response>
"""

context = """
# File: process_list.py.
# takes in a list of integers N: [1, 2, 3, 4]
# and returns the sorted list, while adding 2 to each number and multiplying the result by 5.
# if the resulting number is prime, it outputs 0 instead.
"""

end_context = ""

context = """
The following is a transcript of a rap battle between Nicholas Cage and Donald Trump.
It consists of verses of 8 lines each, with typical Eminem rhyme patterns.
They roast each other by using known facts about each other.
Lyrics:
Verse 1 - Cage
"""

end_context = ""

old_input = ''
print(model('testing'))
# while True:
# with ds_engine:
#     user_input = input('Enter your input: ')
#     prompt = ''
#     if user_input == 'more':
#         prompt = old_input
#     else:
#         prompt = (context + user_input + end_context)
#
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids#.to('cuda') #.to(device)
#
#     gen_tokens = model.generate(
#         input_ids,
#         do_sample=True,
#         # num_beams=3,
#         temperature=0.8,
#         num_beams=2,
#         top_k=50,
#         top_p=0.92,
#         num_return_sequences=1,
#         no_repeat_ngram_size=1,
#         remove_invalid_values=True,
#         max_new_tokens=50,
#     )
#     gen_text = tokenizer.batch_decode(gen_tokens)[0]
#
#     old_input = gen_text
#
#     print(old_input[len(context):])