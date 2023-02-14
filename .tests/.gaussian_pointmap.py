import torch
import torchvision.transforms.functional as F


def generate_point_map(bounding_boxes, height, width, kernel_size=11, sigma=1.0):
    # Initialize a tensor filled with zeros
    map_tensor = torch.zeros((height, width), dtype=torch.float32)

    # Loop over the bounding boxes and place points in the tensor
    for top, left, bb_height, bb_width in bounding_boxes:
        center_y = top + bb_height / 2
        center_x = left + bb_width / 2
        map_tensor[int(center_y), int(center_x)] = 1

    # Perform Gaussian blur on the tensor
    blurred_map = F.gaussian_blur(
        map_tensor.unsqueeze(0).unsqueeze(0),
        (kernel_size, kernel_size),
        sigma
    ).squeeze()

    prev_sum = blurred_map.sum()
    target_sum = len(bounding_boxes)

    # blurred_map = blurred_map / prev_sum * target_sum
    blurred_map = blurred_map * target_sum / prev_sum  # Multiply first to prevent underflow

    assert blurred_map.sum() == target_sum
    return blurred_map


import matplotlib.pyplot as plt

# Define bounding boxes
bounding_boxes = [(10, 10, 20, 20), (30, 30, 10, 10), (50, 50, 30, 30)]
height = 100
width = 100

# Generate the Gaussian map
gaussian_map = generate_point_map(bounding_boxes, height, width)

# Plot the Gaussian map using matplotlib
plt.imshow(gaussian_map, cmap='gray')
plt.savefig(".temp_gaussian.png")
