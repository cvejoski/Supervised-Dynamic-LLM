import json
from pathlib import Path
from typing import List

import torch
from torch.distributions import Gamma
from torch.distributions.transformed_distribution import \
    TransformedDistribution
from torch.distributions.transforms import PowerTransform


def positive_parameter(theta):
    return torch.log(1. + torch.exp(theta))


def positive_parameter_inverse(alpha):
    return torch.log(torch.exp(alpha) - 1.)


def construct_inverse_gamma(a, b):
    """
    If X ~ Gamma(alpha, beta), then 1/X ~ InvGamma(alpha, beta)
    # https://pytorch.org/docs/stable/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution
    """
    power_transform = PowerTransform(-1)
    InverseGamma = TransformedDistribution(Gamma(a, b), transforms=power_transform)
    return InverseGamma


def filter_parameters(param_name: str, allowed_params: List[str]):
    for p in allowed_params:
        if param_name.startswith(p):
            return True
    return False


def load_json(file_name: Path) -> dict:
    assert file_name.suffix == '.json'
    with open(file_name, 'r', encoding='utf-8') as f:
        return json.load(f)
