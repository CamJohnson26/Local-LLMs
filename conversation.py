import os
import torch
from transformers import pipeline, GPTNeoXTokenizerFast, GPTNeoXForCausalLM, Conversation

model_name = 'EleutherAI/pythia-2.8b-deduped'

model = GPTNeoXForCausalLM.from_pretrained(model_name)
tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_name)

pipe = pipeline('conversational', model=model, tokenizer=tokenizer, device=0)

pipe.tokenizer.pad_token_id = model.config.eos_token_id

conversation = None

while True:
    if conversation is None:
        conversation = Conversation(input('Welcome to chat bot:'), )
    else:
        conversation.add_user_input(input())
    conversation = pipe(conversation)
    print(conversation.generated_responses[-1])
