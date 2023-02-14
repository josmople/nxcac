import deeplake
import torchvision.transforms as transforms
import torch

ds = deeplake.load("hub://activeloop/carpk-train")
# ds.create_tensor("boxes", htype="bbox", coords={"type": "fractional", "mode": "LTRB"})


def test(x):

    return x


dl = ds.pytorch(
    num_workers=0, batch_size=1, shuffle=False,
    tensors=["images", "boxes"],
    transform={
        "images": transforms.Compose([
            transforms.ToPILImage(),  # Must convert to PIL image for subsequent operations to run
            transforms.ToTensor(),  # Must convert to pytorch tensor for subsequent operations to run
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]),
        "boxes": test,
    })

for t in dl:
    break

import matplotlib.pyplot as plt

print("Ok")

plt.clf()
img = t["images"][0]
img = (img - img.min()) / (img.max() - img.min())
img = img.permute(1, 2, 0)
plt.imshow(img)
plt.savefig(".temp_01_baseImage.png")


bb = t["boxes"][0]

plt.scatter(bb[:, 0], bb[:, 1])
plt.scatter(bb[:, 0] + bb[:, 2], bb[:, 1] + bb[:, 3])
plt.savefig(".temp_02_withPoints.png")


plt.clf()
cimg = t["images"][0].numpy().permute(1, 2, 0)
for l, t, w, h in bb.int().numpy():
    cimg[t:t + h, l:l + w, :] = 0
plt.imshow(cimg)
plt.savefig(".temp_03_blocked.png")

# CONCLUSION
# Bounding box format is in left, top, width, height
