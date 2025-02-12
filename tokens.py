import os
import urllib.request
import re
from simple_tokenizerV1 import SimpleTokenizerV1
from simple_tokenizerV2 import SimpleTokenizerV2
from gpt_datasetV1 import GPTDatasetV1
from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch

#download the text file
if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)
    
    
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
print("Total number of character:", len(raw_text))
print(raw_text[:99])

text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)

print(result)

result = re.split(r'([,.]|\s)', text)

print(result)

# Strip whitespace from each item and then filter out any empty strings.
result = [item for item in result if item.strip()]
print(result)

text = "Hello, world. Is this-- a test?"

result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])

print(len(preprocessed))


#From these tokens, we can now build a vocabulary that consists of all the unique tokens 

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break


tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))


# text = "Hello, do you like tea. Is this-- a test?"

#tokenizer.encode(text)

# The above produces an error because the word "Hello" is not contained in the vocabulary
# To deal with such cases, we can add special tokens like `"<|unk|>"` to the vocabulary to represent unknown words
# Since we are already extending the vocabulary, let's add another token called `"<|endoftext|>"` which is used in GPT-2 training to denote the end of a text (and it's also used between concatenated text, like if our training datasets consists of multiple articles, books, etc.)

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:integer for integer,token in enumerate(all_tokens)}

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)
    
    
tokenizer2 = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(text)


print(tokenizer2.encode(text))


print(tokenizer2.decode(tokenizer2.encode(text)))

print('##############################################################')



def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)



dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)



print('###############################################################')


input_ids = torch.tensor([2, 3, 5, 1])

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)


print(embedding_layer.weight)

#convert the input_ids to a tensor
print(embedding_layer(torch.tensor([3])))

print(embedding_layer(input_ids))


vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)



#If we sample data from the dataloader, we embed the tokens in each batch into a 256-dimensional vector
#If we have a batch size of 8 with 4 tokens each, this results in a 8 x 4 x 256 tensor:

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)


#  GPT-2 uses absolute position embeddings, so we just create another embedding layer:
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)


# To create the input embeddings used in an LLM, we simply add the token and the positional embeddings:

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
print(input_embeddings)