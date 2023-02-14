import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import bmnet


class NxBmnet(nn.Module):

    def __init__(self, base: bmnet.model.CACModel, counting_features=257, freeze_base=True):
        super().__init__()
        self.base = base
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        self.adjuster = nn.Sequential(
            nn.Conv2d(counting_features, counting_features, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(counting_features, counting_features, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(counting_features, counting_features, 3, padding=1),
        )

    def features(self, images: torch.Tensor, patches: T.Dict[str, torch.Tensor]):
        # Stage 1: extract features for query images and exemplars
        scale_embedding, patches = patches['scale_embedding'], patches['patches']
        features = self.base.backbone(images)
        features = self.base.input_proj(features)

        patches = patches.flatten(0, 1)
        patch_feature = self.base.backbone(patches)  # obtain feature maps for exemplar patches
        patch_feature = self.base.EPF_extractor(patch_feature, scale_embedding)  # compress the feature maps into vectors and inject scale embeddings

        # Stage 2: enhance feature representation, e.g., the self similarity module.
        refined_feature, patch_feature = self.base.refiner(features, patch_feature)
        # Stage 3: generate similarity map by densely measuring similarity.
        counting_feature, corr_map = self.base.matcher(refined_feature, patch_feature)

        return counting_feature

    def find_peaks(self, heatmap: torch.Tensor, k: int = 1) -> torch.Tensor:
        """
        Find the top K peaks in a heatmap tensor.
        :param heatmap: A 4D tensor with shape B x 1 x H x W
        :param K: The number of peaks to find
        :return: A list of peaks, represented as 2D tensors of shape B x 2 (y, x)
        """
        peaks = []
        batch_size, _1, height, width = heatmap.shape

        # loop over each batch
        for b in range(batch_size):
            # Flatten the heatmap and find the indices of the top K values
            heatmap_flat = heatmap[b, 0].flatten()
            _, idx = heatmap_flat.topk(k)
            y, x = idx.div(width), idx.fmod(width)
            peaks.append(torch.stack([y, x], dim=1))

        return peaks

    def avg_point_distance(self, points: torch.Tensor) -> torch.Tensor:
        """
        Calculate the average distance between points in a tensor with shape N x 2.
        :param points: A tensor with shape N x 2
        :return: A scalar representing the average distance between points
        """
        points = points.float()
        num_points = points.shape[0]

        # Calculate the pairwise distances between points using the Euclidean distance formula
        distances = torch.norm(points[:, None] - points, dim=2)

        # Remove the diagonal elements (distances from a point to itself)
        distances = distances[~torch.eye(num_points, dtype=bool).bool()].flatten()

        # Calculate the average distance
        avg_distance = distances.mean()

        return avg_distance

    def negative_mining(self, density: torch.Tensor, image: torch.Tensor, patches: T.Dict[str, torch.Tensor], targets: T.Dict[str, torch.Tensor]):
        neg_patches = {}

        exemplars = patches["patches"]
        scales = patches["scale_embedding"]

        gt_density = targets["density_map"]
        gt_pointmap = targets["pt_map"]
        gt_count = targets["gtcount"]

        from torchvision.transforms.functional import gaussian_blur
        B, P, C, PH, PW = exemplars.shape
        gt_positive_areas = gaussian_blur(gt_pointmap, (PH // 2 * 2 + 1, PH // 2 * 2 + 1))  # Always odd
        gt_negative_mask = (gt_positive_areas < 0.5).float()

        wrong_density = density * gt_negative_mask
        B, _1, H, W = wrong_density.shape
        wrong_density = wrong_density.view(B, _1, H * W).softmax(dim=2).view(B, _1, H, W)

        peaks = self.find_peaks(wrong_density, P)

        gt_avg_point_distances = []
        coord_b, _, coord_y, coord_x = torch.where(gt_pointmap > 0.5)
        for batch_idx in range(B):
            mask = coord_b == batch_idx

            if mask.size(0) == 0:
                gt_avg_point_distances.append(torch.tensor(3, device=mask.device))
                continue

            batch_coord_y = coord_y[mask]
            batch_coord_x = coord_x[mask]
            coords = torch.stack([batch_coord_y, batch_coord_x], dim=1)
            avg_distance = self.avg_point_distance(coords)

            gt_avg_point_distances.append(avg_distance)

        neg_exemplars = []
        for batch_idx in range(B):
            batch_peaks = peaks[batch_idx]
            batch_avg_distance = gt_avg_point_distances[batch_idx]

            half_d = batch_avg_distance // 2
            half_d = half_d.item()

            patches = []

            for cy, cx in batch_peaks.cpu().numpy():
                top = int(np.clip(cy - half_d, 0, H))
                bottom = int(np.clip(cy + half_d, 0, H))
                left = int(np.clip(0, cx - half_d, W))
                right = int(np.clip(0, cx + half_d, W))

                from torchvision.transforms.functional import resize, InterpolationMode
                patch = image[batch_idx, :, top:bottom, left:right]
                patch = resize(patch, (PH, PW), interpolation=InterpolationMode.BILINEAR, antialias=True)

                patches.append(patch)

            while(len(patches) < P):
                patches.append(torch.zeros(C, PH, PW), device=image.device)

            patches = torch.stack(patches, dim=0)
            neg_exemplars.append(patches)

        neg_exemplars = torch.stack(neg_exemplars, dim=1)

        neg_patches["patches"] = neg_exemplars
        neg_patches["scale_embedding"] = scales.float().mean(dim=1, keepdim=True).floor().repeat(1, P).int()

        return neg_patches

    def forward(self, image: torch.Tensor, patches: T.Dict[str, torch.Tensor], targets: T.Dict[str, torch.Tensor]):
        features = self.features(image, patches)
        density = self.base.counter(features)

        neg_patches = self.negative_mining(density, image, patches, targets)
        neg_features = self.features(image, neg_patches)

        neg_features = self.adjuster(neg_features)
        features = features - F.relu(neg_features)

        return self.base.counter(features)


class Adjust(nn.Module):

    def __init__(self, counting_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(counting_features * 2, counting_features * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(counting_features * 2, counting_features, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(counting_features, counting_features, 3, padding=1),
        )

    def forward(self, pos, neg):
        t = torch.cat([pos, neg], dim=1)
        return self.model(t)


class NxBmnet_PNCtx(NxBmnet):

    def __init__(self, base: bmnet.model.CACModel, counting_features=257, freeze_base=True):
        super().__init__(base, counting_features, freeze_base)
        self.base = base
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        self.adjuster = Adjust(counting_features)

    def forward(self, image: torch.Tensor, patches: T.Dict[str, torch.Tensor], targets: T.Dict[str, torch.Tensor]):
        features = self.features(image, patches)
        density = self.base.counter(features)

        neg_patches = self.negative_mining(density, image, patches, targets)
        neg_features = self.features(image, neg_patches)

        neg_features = self.adjuster(neg_features, features)
        features = features - F.relu(neg_features)

        return self.base.counter(features)


class NxBmnet_PNCtx_NegV2(NxBmnet_PNCtx):

    def __init__(self, base: bmnet.model.CACModel, counting_features=257, scale_number=20, freeze_base=True):
        super().__init__(base, counting_features, freeze_base)
        self.scale_number = scale_number

    def negative_mining_item(self, pred_density: torch.Tensor, gt_pointmap: torch.Tensor, boxes: torch.Tensor, exemplar_count=3, distance_threshold=35):
        _1, H, W = pred_density.shape

        from skimage.morphology import dilation, square
        pred_density = dilation(pred_density.squeeze().cpu().detach().numpy() > 0.0001, square(7))
        pred_density = torch.tensor(pred_density, device=gt_pointmap.device)

        background_map = 0.0025 * (1 - pred_density.float())

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

        tlbr[:, 0].clamp_min_(0)
        tlbr[:, 1].clamp_min_(0)
        tlbr[:, 2].clamp_max_(H)
        tlbr[:, 3].clamp_max_(W)

        return tlbr

    def negative_mining(self, density: torch.Tensor, image: torch.Tensor, patches: T.Dict[str, torch.Tensor], targets: T.Dict[str, torch.Tensor], distance_threshold=35):

        neg_patches = {}

        exemplars = patches["patches"]
        scales = patches["scale_embedding"]

        gt_density = targets["density_map"]
        gt_pointmap = targets["pt_map"]
        gt_count = targets["gtcount"]

        tlbrs = targets["tlbr"]

        B, P, C, PH, PW = exemplars.shape
        B, _1, H, W = density.shape

        neg_exemplars = []
        neg_scales = []

        for batch_idx in range(B):
            tlbr = self.negative_mining_item(
                pred_density=density[batch_idx, :, :, :],
                gt_pointmap=gt_pointmap[batch_idx, 0, :, :],
                boxes=tlbrs[batch_idx],
                exemplar_count=P,
                distance_threshold=distance_threshold
            )

            patches = []
            scales = []

            for top, left, bottom, right in tlbr.cpu().numpy():

                from torchvision.transforms.functional import resize, InterpolationMode
                patch = image[batch_idx, :, top:bottom, left:right]
                patch = resize(patch, (PH, PW), interpolation=InterpolationMode.BILINEAR, antialias=True)

                patches.append(patch)
                scale = (right - left) / W * 0.5 + (bottom - top) / H * 0.5
                scale = scale // (0.5 / self.scale_number)
                scale = scale if scale < self.scale_number - 1 else self.scale_number - 1

                scales.append(torch.tensor(scale, dtype=int, device=density.device))

            while(len(patches) < P):
                patches.append(torch.zeros(C, PH, PW, device=image.device))
            while(len(scales) < P):
                scales.append(torch.tensor(0, device=image.device))

            patches = torch.stack(patches, dim=0)
            scales = torch.stack(scales, dim=0)

            neg_exemplars.append(patches)
            neg_scales.append(scales)

        neg_exemplars = torch.stack(neg_exemplars, dim=0)
        neg_scales = torch.stack(neg_scales, dim=0)

        neg_patches["patches"] = neg_exemplars
        neg_patches["scale_embedding"] = neg_scales

        return neg_patches
