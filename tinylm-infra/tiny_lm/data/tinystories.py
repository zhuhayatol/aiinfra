"""
TinyStories dataset (for srs pretraining)
https://huggingface.co/datasets/roneneldan/TinyStories
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python tinystories.py
Will save shards to the directory "../../data/tinystories".

Output:
    ../../data/tinystories/
        tinystories_val_000000.npy
        tinystories_train_000001.npy
        tinystories_train_000002.npy
        ...
"""

import os
import multiprocessing as mp

import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
dataset_path = "roneneldan/TinyStories"
local_dir = "tinystories"
shard_size = int(1e6) # 1M tokens per shard

# 往上俩级找到data目录，并在其下创建新的文件夹
DATA_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), f"../../data/{local_dir}")
)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

print(f"Saving shards to: {DATA_CACHE_DIR}")


# 下载数据集
# fw = load_dataset(dataset_path, split="train")

# 初始化tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token


def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))

    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), \
        "token dictionary too large for uint16"
    
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def process_spilt(dataset, split_name):
    """
    Tokenize one HuggingFace dataset split and save it into shards.

    split_name:
        "train" or "val"
    """
    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count()//2)

    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize, dataset, chunksize=16):

            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"{split_name} Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                filename = os.path.join(DATA_CACHE_DIR, f"tinystories_{split_name}_{shard_index:06d}")
                
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"{split_name} Shard {shard_index}")
     
                progress_bar.update(remainder)
                
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                
                shard_index += 1
                progress_bar.close()
                progress_bar = None
                
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            filename = os.path.join(DATA_CACHE_DIR, f"tinystories_{split_name}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])

            if progress_bar is not None:
                progress_bar.close()

def main():
    train_ds = load_dataset(dataset_path, split="train")
    val_ds   = load_dataset(dataset_path, split="validation")

    process_spilt(train_ds, "train")
    process_spilt(val_ds,   "val")

if __name__ == "__main__":
    main()
