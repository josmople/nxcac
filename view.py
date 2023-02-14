import matplotlib.pyplot as plt
import numpy as np
import torch

import pytorch_utils as U


def heatmap_to_np(heatmap, cmap='jet'):
    figsize = heatmap.shape
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(heatmap, cmap=cmap)
    ax.axis('off')
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def overlay_heatmap(image: torch.Tensor, heatmap: torch.Tensor, filepath, title=None):
    # Convert PyTorch tensors to numpy arrays
    image: np.ndarray = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    heatmap: np.ndarray = heatmap.squeeze(0).squeeze(0).cpu().numpy()

    assert image.ndim == 3
    assert heatmap.ndim == 2

    print(heatmap.shape)

    # Normalize the heatmap to be in the range [0, 1]
    heatmap = heatmap_to_np(heatmap)
    print(heatmap.shape)

    # Overlay the heatmap on the image
    overlaid_image = np.uint8(image * 255)
    overlaid_image = np.uint8(np.minimum((overlaid_image + heatmap) / 2, 255))

    # Plot the overlaid image using Matplotlib
    plt.imshow(overlaid_image)
    plt.axis("off")
    if title is not None:
        plt.title(title)

    # Save the overlaid image to a file
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0)
    plt.close()


def side_comparison(image: torch.Tensor, heatmap: torch.Tensor, filepath, title=None):
    # Convert PyTorch tensors to numpy arrays
    image: np.ndarray = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    heatmap: np.ndarray = heatmap.squeeze(0).squeeze(0).cpu().numpy()

    assert image.ndim == 3
    assert heatmap.ndim == 2

    fig, ax = plt.subplots(1, 2)

    image = (image - image.min()) / (image.max() - image.min())

    h, w = heatmap.shape
    ax[0].imshow(heatmap, cmap="jet", extent=(0, w, h, 0))

    h, w, c = image.shape
    ax[1].imshow(image, extent=(0, w, h, 0))

    if title is not None:
        fig.suptitle(title)

    # Save the overlaid image to a file
    plt.savefig(filepath)
    plt.close()


def viewdir(idx, image: torch.Tensor, heatmap: torch.Tensor, title=None):
    import sys
    import os

    calling_file = sys._getframe(1).f_globals["__file__"]
    name = os.path.splitext(calling_file)[0]

    P = U.dirp.Dirpath(".images", name)

    filepath = P(f"{idx:05}.png")
    side_comparison(image, heatmap, filepath, title)
