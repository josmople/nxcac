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


class CountRegressor(nn.Module):
    def __init__(self, input_channels, pool='mean'):
        super(CountRegressor, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU()
            # nn.LeakyReLU(0.01,inplace=True)
        )

    def forward(self, im):
        # batch_size x num_exemplar x (layers * scales) x H x W
        # batch size here is the number of images you want to count

        # After Regressor
        # num_exemplar x 1 x H x W

        # After Mean/Max
        # 1 x 1 x H x W
        # batch_size x 1 x H x W - so this is basically the number of images and their individual 1xhxw density heat maps

        num_sample = im.shape[0]
        if num_sample == 1:
            output = self.regressor(im.squeeze(0))
            # if eval:
            #     F.leaky_relu(output)
            if self.pool == 'mean':
                output = torch.mean(output, dim=(0), keepdim=True)
                return output
            elif self.pool == 'max':
                output, _ = torch.max(output, 0, keepdim=True)
                return output
        else:
            for i in range(0, num_sample):
                output = self.regressor(im[i])
                if self.pool == 'mean':
                    output = torch.mean(output, dim=(0), keepdim=True)
                elif self.pool == 'max':
                    output, _ = torch.max(output, 0, keepdim=True)

                if i == 0:
                    Output = output
                else:
                    Output = torch.cat((Output, output), dim=0)
            return Output


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


def matlab_gaussian_2d(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return torch.from_numpy(h).float()


def mincount_and_perturbation_loss(densities: torch.Tensor, boxes: torch.Tensor, *, SIGMA=8) -> T.Tuple[torch.Tensor, torch.Tensor]:
    B, _1, H, W = densities.shape
    B, N, _4 = boxes.shape

    mincount_loss = 0.
    perturbation_loss = 0.

    for batch_idx, (density, coords) in enumerate(zip(densities, boxes)):
        for coord in coords:
            t, l, b, r = coord.int().cpu().numpy()
            patch = density[0, t:b, l:r]  # density only has 1 channel

            patch_count = patch.sum()
            _one = torch.ones_like(patch_count, device=densities.device)
            mincount_loss = F.mse_loss(patch_count, _one)

            _gaussian = matlab_gaussian_2d(shape=patch.size(), sigma=SIGMA).to(device=densities.device)  # TODO: sigma is iffy
            perturbation_loss += F.mse_loss(patch, _gaussian)

    return mincount_loss, perturbation_loss


class Famnet(nn.Module):

    def __init__(self, input_channels=6, pool="max", exemplar_scales=[1, 0.9, 1.1], adjustment_steps=100, adjustment_lr=1e-4, weight_mincount=1e-9, weight_perturbation=1e-4):
        super().__init__()
        self.exemplar_scales = exemplar_scales
        self.adjustment_steps = adjustment_steps
        self.adjustment_lr = adjustment_lr
        self.weight_mincount = weight_mincount
        self.weight_perturbation = weight_perturbation

        self.regressor = CountRegressor(input_channels, pool)
        self.feature_extractor = Resnet50FPN()

    def adjustment(self, features: torch.Tensor, tlbrs: torch.Tensor):
        from copy import deepcopy

        features = features.detach()
        tlbrs = tlbrs.detach()

        adapted_regressor = deepcopy(self.regressor)
        adapted_regressor.train()
        optimizer = torch.optim.Adam(adapted_regressor.parameters(), lr=self.adjustment_lr)
        for step in range(self.adjustment_steps):

            output = adapted_regressor(features)

            mincount_loss, perturbation_loss = mincount_and_perturbation_loss(output, tlbrs)
            loss = self.weight_mincount * mincount_loss + self.weight_perturbation * perturbation_loss
            # loss can become zero in some cases, where loss is a 0 valued scalar and not a tensor
            # So Perform gradient descent only for non zero cases
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return adapted_regressor(features)

    def forward(self, images: torch.Tensor, tlbrs: torch.Tensor, adjustment=False):
        B, C, H, W = images.shape
        B, P, _4 = tlbrs.shape

        features = extract_features(self.feature_extractor, images, tlbrs, exemplar_scales=self.exemplar_scales)

        if adjustment:
            return self.adjustment(features, tlbrs)
        return self.regressor(features)
