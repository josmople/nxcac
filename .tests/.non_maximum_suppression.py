import numpy as np
import imageio
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

image = imageio.imread("https://pbs.twimg.com/media/D8Dp0c5WkAAkvME.jpg")
image = np.array(image) / 255

gray = image.mean(axis=2)

plt.clf()
plt.imshow(gray, cmap="gray")
plt.savefig(".temp_01_baseImage")

tensor = torch.tensor(gray)
H, W = tensor.shape
tensor = tensor.view(1, 1, H, W)


tensor = (tensor < 0.5) * 1.0
tensor = TF.gaussian_blur(tensor, (11, 11))

plt.clf()
plt.imshow(tensor.squeeze().numpy(), cmap="gray")
plt.savefig(".temp_02_fakeMask")


kernel_size = 23
pooled_image, _ = F.max_pool2d_with_indices(tensor, kernel_size, stride=1, padding=kernel_size // 2)

plt.clf()
plt.imshow(pooled_image.squeeze().numpy(), cmap="gray")
plt.savefig(".temp_03_maxPoolImage")

mask = pooled_image == tensor

plt.clf()
plt.imshow(mask.squeeze().numpy(), cmap="gray")
plt.savefig(".temp_04_keepMask")

nms_image = mask * tensor

plt.clf()
plt.imshow(nms_image.squeeze().numpy(), cmap="gray")
plt.savefig(".temp_05_nmsImage")

# CONCLUSION
# Removes the non-maximum of a local area
