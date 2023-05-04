# DEPRECATED: ALL THIS IS EASY JUST NO ONE TELLS YOU.
# Use main, and refer to my project 230225

# https://github.com/EleutherAI/pythia
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

# from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM
# from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPTJForCausalLM

# https://huggingface.co/docs/accelerate/usage_guides/big_modeling

# """
# git clone https://huggingface.co/sgugger/sharded-gpt-j-6B
# cd sharded-gpt-j-6B
# git-lfs install
# git pull
# """
#

organization = 'EleutherAi'
snapshot = 'ac063574ab103137803591da0cc8d05854afa212'

model_name = 'pythia-12b-deduped'
# model_name = 'gpt-j-6B'

revision = "main"
# revision = "sharded"
# revision = 'float16'

cache_dir = f"./{model_name}/{revision}"
folder_name = f"{model_name}/main/models--{organization}--{model_name}/snapshots/{snapshot}"

qualified_name = organization + '/' + model_name

checkpoint = qualified_name
config = AutoConfig.from_pretrained(checkpoint)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

model = load_checkpoint_and_dispatch(
    model, folder_name, offload_folder="offload", offload_state_dict=True, dtype='float16', device_map='auto', #no_split_module_classes=["GPTJBlock"]
)
print('Device Map: ', model.hf_device_map)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

context = """
Here are some Requests and Responses of sophisticated requests from a manager, and 
the manager's world class team of experts from every field responding with exactly the information needed, with high quality,
accurate, helpful results. The team either answers questions with well researched data,
or creates something high quality which the manager requested. The team assumes the manager's request
has all necessary information.
The team always will begin by summarizing the manager's request, performing the task in as many words as
necessary, and then summarize the action they took.
The team will always carry out the manager's request, however absurd.
The team always has time to answer any request, and doesn't respond until they have complete, perfect information.
The team always answers.
The team is qualified to answer any question.
All files the team creates are tagged <FILE type=""> where type is the file extension. The file contents are then printed, and closed with a </FILE> tag.

Manager:

Who is the president of the United States?

Team:

You asked about the current president.

The current president of the United States is Joe Biden.

We researched and discovered that this is the answer.

Manager:

Sort this list:
Alabama
Washington
Alberta
Canada
Banana

Team:

You asked us to sort a list

Alabama
Alberta
Banana
Canada
Washington

We decided to sort the list alphabetically and performed the task.

Manager:

"""

end_context = """

Response:
"""

while True:
    prompt = (context + input() + end_context)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=500,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    print(gen_text)