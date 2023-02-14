import bmnet
import matplotlib.pyplot as plt
import numpy as np
import torch
import carpk

if __name__ == "__main__":
    # ds = bmnet.data.build_dataset("D:/Datasets/FSC147_384_V2", "test")
    ds = carpk.build_dataset("test")

    x = ds[0]

    s = x[2]["density_map"].squeeze().numpy()
    s = (s - s.min()) / (s.max() - s.min())
    plt.imshow(s)
    plt.savefig(".temp_01_normDensity.png")

    dx: np.ndarray = np.zeros_like(s)
    tlbr = x[2]["tlbr"]

    for t, l, b, r in tlbr:
        dx[t:b, l:r] = 1

    plt.clf()
    plt.imshow(dx)
    plt.savefig(".temp_02_boxExemplars.png")

    tensor: torch.Tensor = x[0].clone()
    tensor.mul_(torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)).add_(torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1))

    img = tensor.permute(1, 2, 0).numpy()

    img = (img * (1 - dx[:, :, None]) + s[:, :, None]) / 2

    plt.clf()
    plt.imshow(img)
    plt.savefig(".temp_03_visualization.png")

    print()
