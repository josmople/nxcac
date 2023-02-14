import bmnet
from tqdm import tqdm
import torch

import torch.nn.functional as F

import negx.nxbmnet as nxbmnet

import pytorch_utils as U

from glob import glob

device = "cuda"

dataloader = bmnet.data.build_dataloader("D:/Datasets/FSC147_384_V2", "test", batch_size=1)
model = bmnet.model.build_pretrained_model().to(device)


P = U.dirp.Dirpath(".logs", ".baseline")


with torch.no_grad():

    mse = 0
    mae = 0
    tot = 0

    model = nxbmnet.NxBmnet(model, freeze_base=True).to(device)

    for weight_path in glob("D:/Josm/Research/BMNetNegX/.logs/01 - simple/adjuster/*.pth"):
        model.adjuster.load_state_dict(torch.load(weight_path))

        for sample in tqdm(dataloader, dynamic_ncols=True):

            img, patches, targets = sample
            img = img.to(device)
            patches['patches'] = patches['patches'].to(device)
            patches['scale_embedding'] = patches['scale_embedding'].to(device)

            targets['density_map'] = density_map = targets['density_map'].to(device)
            targets['pt_map'] = pt_map = targets['pt_map'].to(device)
            targets['gtcount'] = gtcount = targets["gtcount"].cuda()

            pred_density = model(img, patches, targets)
            # pred_density = model(img, patches, is_train=False)

            error = torch.abs(pred_density.sum() - gtcount).item()
            mae += error
            mse += error ** 2
            tot += 1

        final_mae = mae / tot
        final_mse = (mse / tot) ** 0.5

        print(weight_path)
        print("mae", final_mae)
        print("mse", final_mse)
        print("-" * 100)

        with open("records.txt", "a+") as f:
            f.write("/n".join([
                weight_path,
                f"mae={final_mae}",
                f"mse={final_mse}",
                "/n/n"
            ]))
