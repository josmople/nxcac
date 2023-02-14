import numpy as np
import imageio
import matplotlib.pyplot as plt
import skimage.morphology

image = imageio.imread("https://pbs.twimg.com/media/D8Dp0c5WkAAkvME.jpg")
image = np.array(image) / 255

gray = image.mean(axis=2)

plt.clf()
plt.imshow(gray, cmap="gray")
plt.savefig(".temp_01_baseImage")


mask = gray < 0.5


plt.clf()
plt.imshow(mask, cmap="gray")
plt.savefig(".temp_02_mask")

width = 7
kernel = np.ones((width, width), dtype=float)

density = skimage.morphology.dilation(mask, kernel)

plt.clf()
plt.imshow(density * 1.0, cmap="gray")
plt.savefig(".temp_03_density")

# CONCLUSION
# Expands the mask
