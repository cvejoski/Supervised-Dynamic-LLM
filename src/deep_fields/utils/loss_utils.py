from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
from scipy import sparse
from sklearn import metrics
from torch._six import inf
from torch.distributions import Bernoulli, Beta
from tqdm import tqdm

EPSILON = 1e-10
EULER_GAMMA = 0.5772156649015329


def auc_example():
    beta_prior = Beta(3., 1.)
    beta_sample = beta_prior.sample(sample_shape=(1000,))
    prior_distribution = Bernoulli(beta_sample)
    real_adjacency = prior_distribution.sample()

    pred = torch.zeros_like(beta_sample)
    pred[real_adjacency == 1] = beta_sample[real_adjacency == 1]
    pred[real_adjacency == 0] = 1. - beta_sample[real_adjacency == 0]

    pred = pred.view(-1).detach().cpu().numpy()
    y = real_adjacency.view(-1).cpu().long().numpy()

    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    metrics.auc(fpr, tpr)


def beta_fn(a, b):
    beta_ab = torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))
    return beta_ab


def kullback_leibler_weibull_gamma(k, l, a, b, device, mask=None, reduction='mean'):
    """
     (negative) Kullback-Leibler divergence between Weibull and Gamma distributions:
     k: shape parameter of Weibull distr.
     l: scale parameter of Weibull distr.
     a: shape parameter of Gamma distr.
     b: inverse-scale parameter of Gamma distr.
    """
    epsilon = torch.ones(k.shape).fill_(1e-8).to(device).double()
    if type(a) != torch.Tensor:
        a = torch.ones(k.shape).fill_(a).to(device)
        b = torch.ones(k.shape).fill_(b).to(device)
    else:
        a = a.to(device)
        b = b.to(device)

    k = k.double()
    l = l.double()
    k = torch.max(k, epsilon)
    l = torch.max(l, epsilon)
    kl = -(a * torch.log(l) - np.euler_gamma *
           (a / k) - torch.log(k) - b * l * torch.exp(torch.lgamma(1 + (1 / k))) + np.euler_gamma + 1 + a * torch.log(b) - torch.lgamma(a))

    if reduction == 'mean':
        return torch.mean(kl)
    elif reduction == 'sum':
        return torch.sum(kl)
    else:
        return kl


def log_Bernoulli(x, mean, average=False):
    probs = torch.clamp(mean, min=1e-7, max=1. - 1e-7)
    log_bernoulli = x * torch.log(probs) + (1. - x) * torch.log(1. - probs)
    if average:
        return torch.mean(log_bernoulli)
    else:
        return torch.sum(log_bernoulli)


def lower_likelihood(model, loss, likelihood, KL_theta):
    if loss == inf:
        likelihood = likelihood * 1e-5
        loss = likelihood - model.schedulers["KL_theta"](model.number_of_iterations) * KL_theta
        loss = -torch.mean(loss)
    return loss


def kl_gaussian_diagonal(mu_0, sigma_0, mu_1, sigma_1):
    KL_0 = (sigma_0 / sigma_1)**2
    KL_1 = ((mu_0 - mu_1)**2) / (sigma_1**2)
    KL_2 = 2. * torch.log(sigma_1 / sigma_0)
    KL = .5 * (KL_0 + KL_1 - 1. + KL_2)
    return KL


def kl_gaussian_distributions(mu_0, sigma_0, mu_1, sigma_1):
    KL_0 = (sigma_0 / sigma_1)**2
    KL_1 = ((mu_0 - mu_1)**2) / (sigma_1**2)
    KL_2 = 2. * torch.log(sigma_1 / sigma_0)
    KL = .5 * (KL_0 + KL_1 - 1. + KL_2)
    return KL


def get_doc_freq(data: np.ndarray, wi: int, wj: int = None):
    if wj is None:
        return (data[:, wi] > 0).sum()
    else:
        return sparse.csr_matrix.multiply(data[:, wi] > 0, data[:, wj] > 0).sum()


def kullback_leibler(mean, logvar, reduction='mean'):
    """
    Kullback-Leibler divergence between Gaussian posterior distr.
    with parameters (mean, sigma) and a fixed Gaussian prior
    with mean = 0 and sigma = 1
    """

    kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())  # [B, D]
    skl = torch.sum(kl, dim=-1)
    if reduction == 'mean':
        return torch.mean(skl)
    elif reduction == 'sum':
        return torch.sum(skl)
    else:
        return skl


def kullback_leibler_two_gaussians(mean1, logvar1, mean2, logvar2, reduction='mean'):
    """
    Kullback-Leibler divergence between two Gaussians
    """
    var1 = logvar1.exp()
    var2 = logvar2.exp()
    kl = -0.5 * (1 - logvar2 + logvar1 - ((mean1 - mean2).pow(2) + var1) / var2)  # [B, D]
    skl = torch.sum(kl, dim=-1)
    if reduction == 'mean':
        return torch.mean(skl)
    elif reduction == 'sum':
        return torch.sum(skl)
    else:
        return skl


def kl_kumaraswamy_beta(a, b, prior_alpha, prior_beta, reduction='mean'):
    # compute taylor expansion for E[log (1-v)] term
    # hard-code so we don't have to use Scan()
    kl = 1. / (1 + a * b) * beta_fn(1. / a, b)
    kl += 1. / (2 + a * b) * beta_fn(2. / a, b)
    kl += 1. / (3 + a * b) * beta_fn(3. / a, b)
    kl += 1. / (4 + a * b) * beta_fn(4. / a, b)
    kl += 1. / (5 + a * b) * beta_fn(5. / a, b)
    kl += 1. / (6 + a * b) * beta_fn(6. / a, b)
    kl += 1. / (7 + a * b) * beta_fn(7. / a, b)
    kl += 1. / (8 + a * b) * beta_fn(8. / a, b)
    kl += 1. / (9 + a * b) * beta_fn(9. / a, b)
    kl += 1. / (10 + a * b) * beta_fn(10. / a, b)
    kl *= (prior_beta - 1.) * b

    # use another taylor approx for Digamma function
    # psi_b_taylor_approx = torch.log(b) - 1. / (2 * b) - 1. / (12 * b ** 2)
    psi_b = torch.digamma(b)

    kl += (a - prior_alpha) / a * (-EULER_GAMMA - psi_b - 1. / b)  # T.psi(self.posterior_b)

    # add normalization constants
    kl += torch.log(a + EPSILON) + torch.log(b + EPSILON) + torch.log(beta_fn(prior_alpha, prior_beta) + EPSILON)

    # final term
    kl += -(b - 1) / b

    skl = kl.sum(dim=-1)
    if reduction == 'mean':
        return torch.mean(skl)
    elif reduction == 'sum':
        return torch.sum(skl)
    return skl


def topic_coherence_static(topic_words: np.ndarray, data: np.ndarray, word_counts) -> float:
    D = data.shape[0]
    K = topic_words.shape[0]
    TC = []
    for k in range(K):
        top_10_k = topic_words[k].tolist()
        TC_k = 0
        counter = 0

        for i, word in enumerate(top_10_k):
            # get D(w_i)
            D_wi = word_counts[word]
            j = i + 1
            tmp = 0
            while len(top_10_k) > j > i:
                # get D(w_j) and D(w_i, w_j)
                D_wj = word_counts[top_10_k[j]]
                D_wi_wj = get_doc_freq(data, word, top_10_k[j])
                # get f(w_i, w_j)
                if D_wi_wj == 0:
                    f_wi_wj = -1
                else:
                    f_wi_wj = -1 + (np.log(D_wi) + np.log(D_wj) - 2.0 * np.log(D)) / (np.log(D_wi_wj) - np.log(D))
                # update tmp:
                tmp += f_wi_wj
                j += 1
                counter += 1
            # update TC_k
            TC_k += tmp
        TC.append(TC_k / counter)
        # p_bar.set_postfix_str(f"TC: {np.mean(TC)}")
        # p_bar.update()
    TC = np.mean(TC)
    # print('Topic coherence is: {}'.format(TC))
    return TC


def topic_coherence_dynamic(topic_words: np.ndarray, data: np.ndarray, word_counts, n_worksers: int = 1) -> float:
    tc_all = []
    time_steps = [topic_words[:, i] for i in range(topic_words.shape[1])]

    pool = Pool(n_worksers)
    p_bar = tqdm(pool.imap(partial(topic_coherence_static, data=data, word_counts=word_counts), time_steps, 5),
                 desc="Calculating Topics Coherence:",
                 total=len(time_steps))
    for tt in p_bar:
        tc_all.append(tt)
    pool.close()
    pool.join()

    TC = np.mean(tc_all)
    return TC
