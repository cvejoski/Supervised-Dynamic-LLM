from importlib import import_module

import torch
from torch.distributions import Normal


def kl_householder_flows(z_0, z_last, mean, logvar):
    scale = torch.exp(0.5 * logvar)
    p_0 = Normal(torch.zeros_like(mean), torch.ones_like(mean))
    q_0 = Normal(mean, scale)

    log_q0 = q_0.log_prob(z_0)
    log_p0 = p_0.log_prob(z_last)

    KL = log_q0 - log_p0
    return KL.sum(1).par_1()


def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * (log_var + torch.pow(x - mean, 2) * torch.pow(torch.exp(log_var), -1))
    if average:
        return torch.mean(log_normal, dim)
    else:
        return log_normal


def log_Normal_standard(x, average=False, dim=None):
    log_normal = -0.5 * torch.pow(x, 2)
    if average:
        return torch.mean(log_normal, dim)
    else:
        return log_normal


def get_class_nonlinearity(name):
    """
    Returns non-linearity class (from torch.nn)
    """
    module = import_module("torch.nn")
    clazz = getattr(module, name)

    return clazz


def kullback_leibler(mean, sigma, reduction='mean'):
    """
    Kullback-Leibler divergence between Gaussian posterior distr.
    with parameters (mean, sigma) and a fixed Gaussian prior
    with mean = 0 and sigma = 1
    """
    kl = -0.5 * (1 + 2.0 * torch.log(sigma) - mean * mean - sigma * sigma)  # [B, D]
    skl = torch.sum(kl, dim=1)
    if reduction == 'mean':
        return torch.mean(skl)
    elif reduction == 'sum':
        return torch.sum(skl)
    else:
        return skl


def free_params(module):
    if type(module) is not list:
        module = [module]
    for m in module:
        for p in m.parameters():
            p.requires_grad = True


def frozen_params(module):
    if type(module) is not list:
        module = [module]
    for m in module:
        for p in m.parameters():
            p.requires_grad = False
