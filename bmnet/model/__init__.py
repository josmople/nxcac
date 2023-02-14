import torch

from .backbone import build_backbone
from .counter import get_counter
from .epf_extractor import build_epf_extractor
from .refiner import build_refiner
from .matcher import build_matcher
from .class_agnostic_counting_model import CACModel


def build_model(
    lr_backbone=1e-5,
    backbone="resnet50",
    backbone_layer="layer3",
    fix_bn=True,
    dilation=False,
    epf_extractor="direct_pooling",
    hidden_dim=256,
    repeat_times=1,
    ep_scale_embedding=True,
    ep_scale_number=20,
    refiner="self_similarity_module",
    refiner_proj_dim=32,
    refiner_layers=1,
    matcher="dynamic_similarity_matcher",
    matcher_proj_dim=256,
    dynamic_proj_dim=128,
    use_bias=True,
    counter="density_x16",
    counter_dim=257
):
    backbone = build_backbone(
        lr_backbone=lr_backbone,
        backbone=backbone,
        backbone_layer=backbone_layer,
        fix_bn=fix_bn,
        dilation=dilation,
    )
    epf_extractor = build_epf_extractor(
        epf_extractor=epf_extractor,
        backbone_layer=backbone_layer,
        hidden_dim=hidden_dim,
        repeat_times=repeat_times,
        ep_scale_embedding=ep_scale_embedding,
        ep_scale_number=ep_scale_number

    )
    refiner = build_refiner(
        refiner=refiner,
        hidden_dim=hidden_dim,
        refiner_proj_dim=refiner_proj_dim,
        refiner_layers=refiner_layers
    )
    matcher = build_matcher(
        matcher=matcher,
        hidden_dim=hidden_dim,
        matcher_proj_dim=matcher_proj_dim,
        dynamic_proj_dim=dynamic_proj_dim,
        use_bias=use_bias
    )
    counter = get_counter(
        counter=counter,
        counter_dim=counter_dim
    )

    model = CACModel(backbone, epf_extractor, refiner, matcher, counter, hidden_dim)
    return model


def build_pretrained_model():
    model = build_model(
        lr_backbone=1e-5,
        backbone="resnet50",
        backbone_layer="layer3",
        fix_bn=True,
        dilation=False,
        epf_extractor="direct_pooling",
        hidden_dim=256,
        repeat_times=1,
        ep_scale_embedding=True,
        ep_scale_number=20,
        refiner="self_similarity_module",
        refiner_proj_dim=32,
        refiner_layers=1,
        matcher="dynamic_similarity_matcher",
        matcher_proj_dim=256,
        dynamic_proj_dim=128,
        use_bias=True,
        counter="density_x16",
        counter_dim=257
    )

    from os.path import join, dirname, abspath
    filepath = abspath(join(dirname(__file__), "..", "weights", "model_best.pth"))
    weights = torch.load(filepath)["model"]
    model.load_state_dict(weights)

    return model
