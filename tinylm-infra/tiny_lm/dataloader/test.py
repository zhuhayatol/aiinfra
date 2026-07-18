from dataloader import DataLoaderLite

d = DataLoaderLite(0, 0, 0, 0, split="val", local_dir="tinystories")

d.num_tokens()