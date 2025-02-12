from importlib import metadata
import tiktoken
import torch

from torch.utils.data import Dataset, DataLoader

print("PyTorch version:", torch.__version__)
print("tiktoken version:", metadata.version("tiktoken"))


tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

# BPE tokenizers break down unknown words into subwords and individual characters

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)

strings = tokenizer.decode(integers)

print(strings)




with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(f"x: {x}")
print(f"y: {y}")


for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(context, "---->", desired)
    
    
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

