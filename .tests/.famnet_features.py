import typing as T

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Resnet50FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        children = list(self.resnet.children())
        self.conv1 = nn.Sequential(*children[:4])
        self.conv2 = children[4]
        self.conv3 = children[5]
        self.conv4 = children[6]

    def forward(self, im_data):
        feat_map = self.conv1(im_data)
        feat_map = self.conv2(feat_map)
        feat_map3 = self.conv3(feat_map)
        feat_map4 = self.conv4(feat_map3)
        return feat_map3, feat_map4


@torch.no_grad()
def extract_features(
    feature_extractor: T.Callable[[torch.Tensor], T.List[torch.Tensor]],
    image: torch.Tensor,
    tlbrs: torch.Tensor,
    exemplar_scales=[1.0, 0.9, 1.1],
):
    B, C, H, W = image.shape
    B, P, _4 = tlbrs.shape

    image_features = feature_extractor(image)

    batch_similarities = []

    for batch_idx in range(B):
        tlbr = tlbrs[batch_idx]

        similarities = []  # For each image feature, for each patch scale
        for image_feature in image_features:

            image_feature = image_feature[batch_idx, :, :, :].unsqueeze(0)
            _1, FC, FH, FW = image_feature.shape

            assert H / FH == W / FW
            Scaling = H / FH

            tlbr_scaled = tlbr / Scaling

            tlbr_scaled[:, [0, 1]] = torch.floor(tlbr_scaled[:, [0, 1]])
            tlbr_scaled[:, [2, 3]] = torch.ceil(tlbr_scaled[:, [2, 3]]) + 1  # make the end indices exclusive

            tlbr_scaled[:, [0, 1]] = torch.clamp_min(tlbr_scaled[:, [0, 1]], 0)
            tlbr_scaled[:, 2] = torch.clamp_max(tlbr_scaled[:, 2], FH)
            tlbr_scaled[:, 3] = torch.clamp_max(tlbr_scaled[:, 3], FW)

            patch_heights = tlbr_scaled[:, 2] - tlbr_scaled[:, 0]
            patch_widths = tlbr_scaled[:, 3] - tlbr_scaled[:, 1]

            PH = patch_heights.max().int().item()
            PW = patch_widths.max().int().item()

            patch_features = []

            for patch_idx in range(P):
                top, left = int(tlbr_scaled[patch_idx, 0]), int(tlbr_scaled[patch_idx, 1])
                bottom, right = int(tlbr_scaled[patch_idx, 2]), int(tlbr_scaled[patch_idx, 3])

                patch_feature = image_feature[:, :, top:bottom, left:right]
                patch_feature = F.interpolate(patch_feature, size=(PH, PW), mode='bilinear')
                _1, FC, PH, PW = patch_feature.shape

                patch_features.append(patch_feature)

            patch_features = torch.cat(patch_features, dim=0)
            P, FC, PH, PW = patch_features.shape

            for scale in exemplar_scales:
                PH_scaled = int(np.ceil(PH * scale))
                PW_scaled = int(np.ceil(PW * scale))
                if PH_scaled < 1:  # use original size if scaled size is too small
                    PH_scaled = PH
                if PW_scaled < 1:
                    PW_scaled = PW
                patch_features_scaled = F.interpolate(patch_features, size=(PH_scaled, PW_scaled), mode='bilinear')

                similarity_scaled = F.conv2d(
                    F.pad(image_feature, ((int(PW_scaled / 2)), int((PW_scaled - 1) / 2), int(PH_scaled / 2), int((PH_scaled - 1) / 2))),
                    patch_features_scaled
                )
                similarity_scaled = similarity_scaled.permute([1, 0, 2, 3])

                similarity_scaled = F.interpolate(similarity_scaled, size=(FH, FW), mode='bilinear')
                P, _1, FH, FW = similarity_scaled.shape

                similarities.append(similarity_scaled)

        SH = max([s.size(2) for s in similarities])
        SW = max([s.size(3) for s in similarities])
        similarities = list(map(lambda s: F.interpolate(s, size=(SH, SW), mode='bilinear'), similarities))

        similarities = torch.cat(similarities, dim=1)
        P, S, SH, SW = similarities.shape
        assert S == len(exemplar_scales) * len(image_features)

        batch_similarities.append(similarities)

    batch_similarities = torch.stack(batch_similarities, dim=0)
    B, P, S, SH, SW = batch_similarities.shape
    return batch_similarities


model = Resnet50FPN()


features = extract_features(
    feature_extractor=model,
    image=torch.ones(1, 3, 600, 600),
    tlbrs=torch.as_tensor([
        [126, 251, 189, 296],
        [36, 293, 114, 345],
        [107, 475, 159, 542]
    ]).view(1, 3, 4)
)


print()
