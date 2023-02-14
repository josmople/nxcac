import bmnet
from tqdm import tqdm
import torch

import torch.nn.functional as F

import famnet

import pytorch_utils as U

from glob import glob

device = "cuda"

dataloader = bmnet.data.build_dataloader("D:/Datasets/FSC147_384_V2", "test", batch_size=1)
model = famnet.build_pretrained_model().to(device).eval()


mse = 0
mae = 0
tot = 0

for sample in tqdm(dataloader, dynamic_ncols=True):

    img, patches, targets = sample
    img = img.to(device)

    gtcount = targets["gtcount"].cuda()
    tlbrs = targets["tlbr"]

    # pred_density = model(img, patches, targets)
    pred_density = model(img, tlbrs, adjustment=True)

    error = torch.abs(pred_density.sum() - gtcount).item()
    mae += error
    mse += error ** 2
    tot += 1

final_mae = mae / tot
final_mse = (mse / tot) ** 0.5

print("mae", final_mae)
print("mse", final_mse)
print("-" * 100)
