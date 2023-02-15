import bmnet
from tqdm import tqdm
import torch

import torch.nn.functional as F

import negx.nxbmnet as nxbmnet

import pytorch_utils as U


device = "cuda"

dataloader = bmnet.data.build_dataloader("D:\Datasets\FSC147_384_V2", "train", batch_size=1)
model = bmnet.model.build_pretrained_model().to(device)

model = nxbmnet.NxBmnet_PNCtx(model, freeze_base=True)
model.base.counter = bmnet.model.get_counter(counter="density_x16", counter_dim=257)
model.to(device)

model.adjuster.load_state_dict(torch.load("D:/Josm/Research/BMNetNegX/.logs/02 - trainable counter/adjuster/072.pth"))
model.base.counter.load_state_dict(torch.load("D:/Josm/Research/BMNetNegX/.logs/02 - trainable counter/counter/072.pth"))


optim = torch.optim.Adam(list(model.adjuster.parameters()) + list(model.base.counter.parameters()), lr=1e-7)

P = U.dirp.Dirpath(".logs", "02 - trainable counter (p2)")

writer = U.log.tb_writer(P())


for epoch in range(100):

    print("Epoch", epoch)

    mse = 0
    mae = 0
    tot = 0

    for sample in tqdm(dataloader, dynamic_ncols=True):

        img, patches, targets = sample
        img = img.to(device)
        patches['patches'] = patches['patches'].to(device)
        patches['scale_embedding'] = patches['scale_embedding'].to(device)

        targets['density_map'] = density_map = targets['density_map'].to(device)
        targets['pt_map'] = pt_map = targets['pt_map'].to(device)
        targets['gtcount'] = gtcount = targets["gtcount"].cuda()

        pred_density = model(img, patches, targets)
        loss = F.mse_loss(pred_density, density_map)

        optim.zero_grad()
        loss.backward()
        optim.step()

        error = torch.abs(pred_density.sum() - gtcount).item()
        mae += error
        mse += error ** 2
        tot += 1

    final_mae = mae / tot
    final_mse = (mse / tot) ** 0.5

    torch.save(model.adjuster.state_dict(), P("adjuster", f"{epoch:03}.pth"))
    torch.save(model.base.counter.state_dict(), P("counter", f"{epoch:03}.pth"))

    print("mse", final_mae)
    print("mae", final_mse)

    writer.add_scalar("mae", final_mae, epoch)
    writer.add_scalar("mse", final_mse, epoch)
