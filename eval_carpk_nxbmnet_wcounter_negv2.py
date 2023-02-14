import bmnet
from tqdm import tqdm
import torch

import torch.nn.functional as F

import carpk
import negx.nxbmnet as nxbmnet

import pytorch_utils as U

from glob import glob

import view

device = "cuda"

# dataloader = carpk.build_dataloader("test", batch_size=1)
dataloader = carpk.build_cached_dataloader(".temp/carpk_cache", batch_size=1)

model = bmnet.model.build_pretrained_model().to(device)


# model = adapter.NegXBmnet(model, freeze_base=True).to(device)


model = nxbmnet.NxBmnet_PNCtx_NegV2(model, freeze_base=True).to(device)
model.adjuster.load_state_dict(torch.load("D:/Josm/Research/BMNetNegX/.logs/03 - alt negative mining/adjuster/045.pth"))
model.base.counter.load_state_dict(torch.load("D:/Josm/Research/BMNetNegX/.logs/03 - alt negative mining/counter/045.pth"))


with torch.no_grad():

    mse = 0
    mae = 0
    tot = 0

    for i, sample in enumerate(tqdm(dataloader, dynamic_ncols=True)):

        img, patches, targets = sample
        img = img.to(device)
        patches['patches'] = patches['patches'].to(device)
        patches['scale_embedding'] = patches['scale_embedding'].to(device)

        targets['density_map'] = density_map = targets['density_map'].to(device)
        targets['pt_map'] = pt_map = targets['pt_map'].to(device)
        targets['gtcount'] = gtcount = targets["gtcount"].cuda()

        pred_density = model(img, patches, targets)
        # pred_density = model(img, patches, is_train=False)
        view.viewdir(i, img, pred_density, f"pred={pred_density.sum().cpu().item():.2f}, gt={gtcount.cpu().item()}")

        error = torch.abs(pred_density.sum() - gtcount).item()
        mae += error
        mse += error ** 2
        tot += 1

    final_mae = mae / tot
    final_mse = (mse / tot) ** 0.5

    print("mae", final_mae)
    print("mse", final_mse)
    print("-" * 100)
