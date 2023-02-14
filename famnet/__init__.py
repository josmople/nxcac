from . import model


def build_pretrained_model(pool="max", exemplar_scales=[1.0, 0.9, 1.1], adjustment_steps=100, adjustment_lr=1e-4, weight_mincount=1e-9, weight_perturbation=1e-4):
    import torch

    input_channels = 2 * len(exemplar_scales)
    m = model.Famnet(input_channels, pool, exemplar_scales)

    from os.path import join, dirname

    m.regressor.load_state_dict(torch.load(join(dirname(__file__), "weights", "famnet_counter.pth")))

    return m
