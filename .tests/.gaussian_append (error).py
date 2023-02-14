import torch
import torch.nn.functional as F


def generate_gaussian_map(bboxes, height, width, sigma=1.0):
    """
    Generates a 2D Gaussian map using the given bounding boxes.

    Parameters:
        bboxes (list): List of bounding boxes in format of (top, left, height, width).
        height (int): Height of the 2D Gaussian map.
        width (int): Width of the 2D Gaussian map.
        sigma (float, optional): Standard deviation of the Gaussian distribution. Defaults to 1.0.

    Returns:
        torch.Tensor: 2D Gaussian map with shape (1, 1, height, width).
    """
    gaussian_map = torch.zeros((1, 1, height, width), dtype=torch.float32)

    for bbox in bboxes:
        top, left, h, w = bbox
        bottom, right = top + h, left + w
        center_y, center_x = top + h / 2, left + w / 2

        # Create meshgrid
        x = torch.arange(0, w, dtype=torch.float32)
        y = torch.arange(0, h, dtype=torch.float32)
        X, Y = torch.meshgrid(x, y)

        # Calculate distances from the center of the bounding box
        y_distance = Y - center_y
        x_distance = X - center_x

        # Square the distances and divide by 2 * sigma ** 2
        squared_distance = (y_distance ** 2 + x_distance ** 2) / (2 * sigma ** 2)

        # Calculate the 2D Gaussian distribution
        gaussian = torch.exp(-squared_distance)

        gaussian = gaussian / gaussian.sum()  # Sum must be equal to 1

        gaussian_map[0, 0, top:bottom, left:right] = gaussian

    return gaussian_map


import matplotlib.pyplot as plt
import numpy as np

# Call the generate_gaussian_map function
bboxes = [(100, 100, 10, 10), (200, 200, 10, 10), (300, 300, 10, 10)]
height, width = 512, 512
gaussian_map = generate_gaussian_map(bboxes, height, width, sigma=1.0)

# Convert the tensor to a numpy array
gaussian_map = gaussian_map.numpy().squeeze()

# Plot the 2D Gaussian map using Matplotlib
plt.imshow(gaussian_map, cmap='gray')
plt.savefig(".temp_gaussian.png")

# CONCLUSION
# There are zeroes appearing in the final gaussian tensor
