import numpy as np
import imageio
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

image = imageio.imread("https://pbs.twimg.com/media/D8Dp0c5WkAAkvME.jpg")
image = np.array(image) / 255

# See .non_maximum_suppression.py for code explanation
gray = image.mean(axis=2)
tensor = torch.tensor(gray)
H, W = tensor.shape
tensor = tensor.view(1, 1, H, W)
tensor = (tensor < 0.5) * 1.0
tensor = TF.gaussian_blur(tensor, (11, 11))
kernel_size = 23
pooled_image, _ = F.max_pool2d_with_indices(tensor, kernel_size, stride=1, padding=kernel_size // 2)
mask = pooled_image == tensor
nms_image = mask * tensor


x, y = torch.where(nms_image.squeeze() > 0)
xy = torch.stack([x, y], dim=1)

print()
# CONCLUSION
# torch.where returns the coordinates for non-zero items
