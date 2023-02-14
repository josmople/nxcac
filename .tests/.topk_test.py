import torch

K = 0.5
scores = torch.randn(1, 1, 4, 5)

batch, cat, height, width = scores.size()
assert cat == 1

inds_all = torch.where((scores.view(batch, cat, -1) >= K))
inds = inds_all[2]


inds = torch.reshape(inds, (1, 1, inds.shape[0]))

inds_adj = inds % (height * width)
y_coords = (inds_adj / width).int().float()
x_coords = (inds_adj % width).int().float()


mx = x_coords.shape[2]
my = y_coords.shape[2]

x_coords = torch.reshape(x_coords, (mx, 1))
y_coords = torch.reshape(y_coords, (my, 1))
peaks = torch.cat((x_coords, y_coords), 1)  # X, Y


# Alternate approach

coords = torch.where(scores >= K)  # B, C, Y, X
test = torch.stack(coords, dim=1)[:, [3, 2]]


def to_sorted(t: torch.Tensor):
    nums = t[:, 0] * 100 + t[:, 1]
    nums, _ = nums.sort()
    return nums


keys1 = to_sorted(peaks)
keys2 = to_sorted(test)

print(keys1 == keys2)

# CONLUSION
# The approaches are same
