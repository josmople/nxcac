import typing as T
import deeplake

import torch
import torchvision.transforms.functional as TF

import os
import pytorch_utils as U


"""
FSC-147 dataset
The exemplar boxes are sampled and resized to the same size
"""
from torch.utils.data import Dataset
import os
from PIL import Image
import json
import torch
import numpy as np
from torchvision.transforms import transforms

import matplotlib.pyplot as plt


def get_image_classes(class_file):
    class_dict = dict()
    with open(class_file, 'r') as f:
        classes = [line.split('\t') for line in f.readlines()]

    for entry in classes:
        class_dict[entry[0]] = entry[1]

    return class_dict


def batch_collate_fn(batch):
    batch = list(zip(*batch))
    batch[0], scale_embedding, batch[2] = batch_padding(batch[0], batch[2])
    patches = torch.stack(batch[1], dim=0)
    batch[1] = {'patches': patches, 'scale_embedding': scale_embedding.long()}
    return tuple(batch)


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def batch_padding(tensor_list, target_dict):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        density_shape = [len(tensor_list)] + [1, max_size[1], max_size[2]]
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        density_map = torch.zeros(density_shape, dtype=dtype, device=device)
        pt_map = torch.zeros(density_shape, dtype=dtype, device=device)
        gtcount = []
        scale_embedding = []
        tlbrs = []
        for idx, package in enumerate(zip(tensor_list, tensor, density_map, pt_map)):
            img, pad_img, pad_density, pad_pt_map = package
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            pad_density[:, : img.shape[1], : img.shape[2]].copy_(target_dict[idx]['density_map'])
            pad_pt_map[:, : img.shape[1], : img.shape[2]].copy_(target_dict[idx]['pt_map'])
            gtcount.append(target_dict[idx]['gtcount'])
            scale_embedding.append(target_dict[idx]['scale_embedding'])
            tlbrs.append(target_dict[idx]['tlbr'])
        target = {
            'density_map': density_map,
            'pt_map': pt_map,
            'gtcount': torch.tensor(gtcount),
            'tlbr': torch.stack(tlbrs, dim=0)
        }
    else:
        raise ValueError('not supported')
    return tensor, torch.stack(scale_embedding), target


class CarpkDataset(Dataset):
    def __init__(
        self, kind="train", scaling=1.0, box_number=3, scale_number=20, min_size=384, max_size=1584, preload=False, main_transform=None, query_transform=None,
        access_method="local:1", cache=".temp/carpk"
    ):
        os.makedirs(cache, exist_ok=True)
        os.environ["DEEPLAKE_DOWNLOAD_PATH"] = cache
        self.ds = deeplake.load(f"hub://activeloop/carpk-{kind}", access_method=access_method)

        self.scaling = scaling
        self.box_number = box_number
        self.scale_number = scale_number
        self.preload = preload
        self.main_transform = main_transform
        self.query_transform = query_transform
        self.min_size = min_size
        self.max_size = max_size

        # store images and generate ground truths
        self.images = {}
        self.targets = {}
        self.patches = {}

    def generate_density(self, H, W, tlhws: np.ndarray, sigma=1):
        map_tensor = np.zeros((H, W), float)

        for top, left, bb_height, bb_width in tlhws:
            center_y = top + bb_height / 2
            center_x = left + bb_width / 2
            map_tensor[int(center_y), int(center_x)] = 1

        tensor = torch.tensor(map_tensor).view(1, 1, H, W)

        avg_height = int(tlhws[:, 2].mean() // 2 * 2) + 1
        avg_width = int(tlhws[:, 3].mean() // 2 * 2) + 1

        tensor = TF.gaussian_blur(tensor, (avg_height, avg_width), sigma=10)
        tensor = tensor.view(H, W).numpy()

        target_sum = tlhws.shape[0]
        prev_sum = tensor.sum()
        tensor = tensor * target_sum / prev_sum

        return tensor

    def generate_points(self, tlhws: np.ndarray):
        centers = tlhws[:, [0, 1]] + tlhws[:, [2, 3]] / 2
        return np.rint(centers).astype(int)

    def generate_boxes(self, tlhws: np.ndarray):
        top_left = tlhws[:, [0, 1]]

        top_right = top_left.copy()
        top_right[:, 1] += tlhws[:, 3]

        bottom_right = top_left.copy()
        bottom_right += tlhws[:, [2, 3]]

        bottom_left = top_left.copy()
        bottom_left[:, 0] += tlhws[:, 2]

        return np.stack([top_left, top_right, bottom_right, bottom_left], axis=1)[:, :, [1, 0]]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):

        if idx in self.images:
            img = self.images[idx]
            target = self.targets[idx]
            patches = self.patches[idx]

        else:
            # density_path = os.path.join(self.data_dir, 'gt_density_map_adaptive_384_VarV2/' + file_name.replace('jpg', 'npy'))

            # img_info = self.annotations[file_name]
            img = Image.fromarray(self.ds[idx]["images"].numpy())
            w, h = img.size
            # resize the image
            r = 1.0
            if h > self.max_size or w > self.max_size:
                r = self.max_size / max(h, w)
            if r * h < self.min_size or w * r < self.min_size:
                r = self.min_size / min(h, w)
            nh, nw = int(r * h), int(r * w)
            img = img.resize((nw, nh), resample=Image.BICUBIC)

            ltwhs = self.ds[idx]["boxes"].numpy()
            tlhws = ltwhs[:, [1, 0, 3, 2]]  # Tech debt, made a mistake

            #target = np.zeros((nh, nw), dtype=np.float32)
            # density_map = np.load(density_path).astype(np.float32)
            density_map = self.generate_density(h, w, tlhws)
            pt_map = np.zeros((nh, nw), dtype=np.int32)
            # points = (np.array(img_info['points']) * r).astype(np.int32)
            points = self.generate_points(tlhws)
            # boxes = np.array(img_info['box_examples_coordinates']) * r
            boxes = self.generate_boxes(tlhws) * r
            boxes = boxes[:self.box_number, :, :]
            gtcount = points.shape[0]

            # crop patches and data transformation
            target = dict()
            patches = []
            scale_embedding = []

            #print('boxes:', boxes.shape[0])
            if points.shape[0] > 0:
                points[:, 0] = np.clip(points[:, 0], 0, nw - 1)
                points[:, 1] = np.clip(points[:, 1], 0, nh - 1)
                pt_map[points[:, 1], points[:, 0]] = 1
                for box in boxes:
                    x1, y1 = box[0].astype(np.int32)
                    x2, y2 = box[2].astype(np.int32)
                    patch = img.crop((x1, y1, x2, y2))
                    patches.append(self.query_transform(patch))
                    # calculate scale
                    scale = (x2 - x1) / nw * 0.5 + (y2 - y1) / nh * 0.5
                    scale = scale // (0.5 / self.scale_number)
                    scale = scale if scale < self.scale_number - 1 else self.scale_number - 1
                    scale_embedding.append(scale)

            target['density_map'] = density_map * self.scaling
            target['pt_map'] = pt_map
            target['gtcount'] = gtcount
            target['scale_embedding'] = torch.tensor(scale_embedding)

            tlbr = []
            for box in boxes:
                l, t = box[0].astype(np.int32)
                r, b = box[2].astype(np.int32)
                tlbr.append((t, l, b, r))
            target["tlbr"] = torch.tensor(tlbr)

            img, target = self.main_transform(img, target)
            patches = torch.stack(patches, dim=0)

            if self.preload:
                self.images.update({idx: img})
                self.patches.update({idx: patches})
                self.targets.update({idx: target})

        return img, patches, target


def pad_to_constant(inputs, psize):
    h, w = inputs.size()[-2:]
    ph, pw = (psize - h % psize), (psize - w % psize)
    # print(ph,pw)

    (pl, pr) = (pw // 2, pw - pw // 2) if pw != psize else (0, 0)
    (pt, pb) = (ph // 2, ph - ph // 2) if ph != psize else (0, 0)
    if (ph != psize) or (pw != psize):
        tmp_pad = [pl, pr, pt, pb]
        # print(tmp_pad)
        inputs = torch.nn.functional.pad(inputs, tmp_pad)

    return inputs


def pad_to_constant_padding_values(inputs, psize):
    h, w = inputs.size()[-2:]
    ph, pw = (psize - h % psize), (psize - w % psize)
    # print(ph,pw)

    (pl, pr) = (pw // 2, pw - pw // 2) if pw != psize else (0, 0)
    (pt, pb) = (ph // 2, ph - ph // 2) if ph != psize else (0, 0)
    if (ph != psize) or (pw != psize):
        return [pl, pr, pt, pb]

    return [0, 0, 0, 0]


class MainTransform(object):
    def __init__(self):
        self.img_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __call__(self, img, target):
        img = self.img_trans(img)
        density_map = target['density_map']
        pt_map = target['pt_map']
        pt_map = torch.from_numpy(pt_map).unsqueeze(0)
        density_map = torch.from_numpy(density_map).unsqueeze(0)

        img = pad_to_constant(img, 32)
        density_map = pad_to_constant(density_map, 32)
        pt_map = pad_to_constant(pt_map, 32)
        target['density_map'] = density_map.float()
        target['pt_map'] = pt_map.float()

        pl, pr, pt, pb = pad_to_constant_padding_values(img, 32)
        tlbr: torch.Tensor = target["tlbr"].clone()
        tlbr[:, 0] += pt
        tlbr[:, 1] += pl
        tlbr[:, 2] += pt
        tlbr[:, 3] += pl
        target["tlbr"] = tlbr.int()

        return img, target


def get_query_transforms(augment, exemplar_size):
    if augment:
        # SimCLR style augmentation
        return transforms.Compose([
            transforms.Resize(exemplar_size),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),  HorizontalFlip may cause the pretext too difficult, so we remove it
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(exemplar_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def build_dataset(kind="test", positive_exemplars=3, exemplar_size=(128, 128), ep_scale_number=20, with_augmentations=False, access_method="local:1", cache=".temp/carpk"):
    main_transform = MainTransform()
    query_transform = get_query_transforms(with_augmentations, exemplar_size)

    dataset = CarpkDataset(
        kind=kind,
        scaling=1.0,
        box_number=positive_exemplars,
        scale_number=ep_scale_number,
        main_transform=main_transform,
        query_transform=query_transform,
        access_method=access_method,
        cache=cache
    )

    return dataset


def build_dataloader(kind="test", batch_size=8, positive_exemplars=3, exemplar_size=(128, 128), ep_scale_number=20, with_augmentations=False, access_method="local:1", cache=".temp/carpk", **kwds):
    dataset = build_dataset(
        kind=kind,
        positive_exemplars=positive_exemplars,
        exemplar_size=exemplar_size,
        ep_scale_number=ep_scale_number,
        with_augmentations=with_augmentations,
        access_method=access_method,
        cache=cache
    )

    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, collate_fn=batch_collate_fn, **kwds)


def build_cache(path, **kwds):
    from tqdm import tqdm
    P = U.dirp.Dirpath(path)

    ds = build_dataset(**kwds)

    for i in tqdm(range(len(ds)), dynamic_ncols=True):

        filepath = P(f"{i:05}.pt")
        if os.path.exists(filepath):
            continue

        item = ds[i]
        torch.save(item, filepath)


def build_cached_dataset(path):
    return U.data.glob_files(os.path.join(path, "*.pt"), torch.load)


def build_cached_dataloader(path, batch_size=8, **kwds):
    ds = build_cached_dataset(path)

    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch_size, collate_fn=batch_collate_fn, **kwds)


if __name__ == "__main__":
    ds = build_dataset("test")
    item = ds[0]

    print(item)
