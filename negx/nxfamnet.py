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


class SimKernel(nn.Module):
    def __init__(self, input_channels, n_feats=2):
        super(SimKernel, self).__init__()
        self.kernel = nn.ModuleList([
            nn.Conv2d(input_channels[i], input_channels[i], kernel_size=1, stride=1)
            for i in range(n_feats)
        ])

    def forward0(self, images):
        for index in range(len(self.kernel)):
            yield self.kernel[index](images[index])

    def forward(self, images: T.List[torch.Tensor]):
        return tuple(self.forward0(images))


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


class NxFamnet(nn.Module):

    def __init__(self, input_channels=6, pool="max", exemplar_scales=[1, 0.9, 1.1]):
        super().__init__()
        self.exemplar_scales = exemplar_scales

        self.regressor = CountRegressor(input_channels, pool)

        self.pos_kern = SimKernel([512, 1024])
        self.neg_kern = SimKernel([512, 1024])

        self.feature_extractor = Resnet50FPN()

    def positive_features(self, tensor, tlbrs):
        def feats(tensor):
            return self.pos_kern(self.feature_extractor(tensor))
        return extract_features(feats, tensor, tlbrs, scales=self.exemplar_scales)

    def negative_features(self, tensor, tlbrs):
        def feats(tensor):
            return self.neg_kern(self.feature_extractor(tensor))
        return extract_features(feats, tensor, tlbrs, scales=self.exemplar_scales)

    def negative_mining_item(self, pred_density: torch.Tensor, gt_pointmap: torch.Tensor, boxes: torch.Tensor, exemplar_count=3, distance_threshold=35):
        _1, H, W = pred_density.shape

        from skimage.morphology import dilation, square
        pred_density = dilation(pred_density.squeeze() > 0.0001, square(7))
        pred_density = torch.tensor(pred_density)

        background_map = 0.0025 * (1 - pred_density)

        def non_maximum_suppression(heat: torch.Tensor, kernel=3):
            pad = (kernel - 1) // 2
            hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
            keep = (hmax == heat).float()
            return heat * keep  # filters to keep the highest values only
        heat = non_maximum_suppression(background_map.view(1, 1, H, W))

        k = 0.0025

        def threshold_filter(heat: torch.Tensor, K):
            coords = torch.where(heat >= K)  # B, C, Y, X
            return torch.stack(coords, dim=1)[:, [2, 3]]
        background_peaks = threshold_filter(heat, K=k)

        if len(background_peaks) <= 0:
            return torch.tensor([], device=pred_density.device)

        coord_y, coord_x = torch.where(gt_pointmap > 0.5)
        gt_points = torch.stack([coord_y, coord_x], dim=1)

        pairwise_distance = torch.cdist(gt_points.double(), background_peaks.double())

        peak_filter = pairwise_distance.min(dim=0).values > distance_threshold
        far_background_peaks = background_peaks[peak_filter]

        random_indexes = torch.randperm(far_background_peaks.shape[0])[:exemplar_count]
        far_background_peaks = far_background_peaks[random_indexes]

        # def avg_point_distance(points: torch.Tensor) -> torch.Tensor:
        #     points = points.float()
        #     num_points = points.shape[0]
        #     distances = torch.norm(points[:, None] - points, dim=2)
        #     distances = distances[~torch.eye(num_points, dtype=bool).bool()].flatten()
        #     avg_distance = distances.mean()
        #     return avg_distance
        # avg_distance = avg_point_distance(gt_pointmap)
        # half_distance = int(avg_distance / 2)

        def avg_height_width(boxes):
            height = boxes[:, 2] - boxes[:, 0]
            width = boxes[:, 3] - boxes[:, 1]
            avg_height = height.float().mean()
            avg_width = width.float().mean()
            return avg_height.int(), avg_width.int()

        avg_height, avg_width = avg_height_width(boxes)
        half_height, half_width = int(avg_height // 2), int(avg_width // 2)

        tlbr = far_background_peaks[:, [0, 1, 0, 1]].contiguous()
        tlbr[:, 0] -= half_height
        tlbr[:, 1] -= half_width
        tlbr[:, 2] += half_height
        tlbr[:, 3] += half_width

        # tlbr = far_background_peaks[:, [0, 1, 0, 1]].contiguous()
        # tlbr[:, 0] -= half_distance
        # tlbr[:, 1] -= half_distance
        # tlbr[:, 2] += half_distance
        # tlbr[:, 3] += half_distance

        tlbr[:, 0].clamp_min_(0)
        tlbr[:, 1].clamp_min_(0)
        tlbr[:, 2].clamp_max_(H)
        tlbr[:, 3].clamp_max_(W)

        return tlbr

    def negative_mining(self, pred_density: torch.Tensor, gt_pointmap: torch.Tensor, boxes: torch.Tensor, exemplar_count=3, distance_threshold=35):
        B, _1, H, W = pred_density.shape
        tlbrs = []
        for batch_idx in range(B):
            tlbr = self.negative_mining_item(
                pred_density=pred_density[batch_idx, :, :, :],
                gt_pointmap=gt_pointmap[batch_idx, 0, :, :],
                boxes=boxes,
                exemplar_count=exemplar_count,
                distance_threshold=distance_threshold
            )
            tlbrs.append(tlbr)
        return torch.stack(tlbrs, dim=0)

    def forward(self, images: torch.Tensor, tlbrs: torch.Tensor, gt_pointmap: torch.Tensor):
        B, C, H, W = images.shape
        B, P, _4 = tlbrs.shape
        B, _1, H, W = gt_pointmap

        features = self.positive_features(images, tlbrs)
        pred_density = self.regressor(features)

        neg_tlbrs = self.negative_mining(pred_density=pred_density, gt_pointmap=gt_pointmap, boxes=tlbrs, num_tlbrs=P)
        neg_features = self.negative_features(images, neg_tlbrs, self.neg_kern)

        features = features - F.relu(neg_features)
        return self.regressor(features)


def build_pretrained_model(pool="max", exemplar_scales=[1.0, 0.9, 1.1]):

    input_channels = 2 * len(exemplar_scales)
    model = NxFamnet(input_channels, pool, exemplar_scales)

    from os.path import join, dirname

    model.regressor.load_state_dict(torch.load(join(dirname(__file__), "weights", "nxfamnet_counter.pth")))
    model.pos_kern.load_state_dict(torch.load(join(dirname(__file__), "weights", "nxfamnet_pos_kern.pth")))
    model.neg_kern.load_state_dict(torch.load(join(dirname(__file__), "weights", "nxfamnet_neg_kern.pth")))

    return model
