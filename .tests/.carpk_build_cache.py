import carpk
from tqdm import tqdm

import pytorch_utils as U
import torch

import os

P = U.dirp.Dirpath(".temp", "carpk_cache")

ds = carpk.build_dataset("test")

for i in tqdm(range(len(ds)), dynamic_ncols=True):

    filepath = P(f"{i:05}.pt")
    if os.path.exists(filepath):
        continue

    item = ds[i]
    torch.save(item, filepath)
