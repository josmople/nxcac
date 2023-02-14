import negx
import torch

model = negx.nxfamnet.build_pretrained_model()

regressor_sd = torch.load("D:/Josm/Research/BMNetNegX/negx/weights/FamNet_kern_count_best.pth")
pos_kernel_sd = torch.load("D:/Josm/Research/BMNetNegX/negx/weights/FamNet_kern_pos_best.pth")


model.regressor.load_state_dict(regressor_sd)
model.pos_kern.load_state_dict(pos_kernel_sd)

print
