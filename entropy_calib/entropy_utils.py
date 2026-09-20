"""KDE-based predictive-entropy estimator.

Ported verbatim from DemoSpeedup's
`aloha/act/detr/models/entropy_utils.py` (github.com/lingxiao-guo/DemoSpeedup).
No ACT/Aloha dependency in the original file, so nothing was changed here.
"""

import torch
from scipy.special import digamma


def k_nn_distance(x, k):
    batch_size, num_samples, dim = x.size()

    x_flat = x.view(batch_size, num_samples, -1)
    distances = torch.cdist(x_flat, x_flat)  # (batch_size, num_samples, num_samples)

    k_distances, _ = torch.topk(distances, k + 1, dim=-1, largest=False)
    k_distances = k_distances[:, :, 1:]

    return k_distances


def kozachenko_leonenko_entropy(x, k=5):
    batch_size, num_samples, dim = x.size()

    k_distances = k_nn_distance(x, k)

    avg_distances = k_distances.mean(dim=2)

    digamma_k = torch.tensor(digamma(k), dtype=torch.float32, device=x.device)
    digamma_n = torch.tensor(digamma(num_samples), dtype=torch.float32, device=x.device)

    entropy = digamma_n - digamma_k - dim * torch.log(avg_distances).mean(dim=1, keepdim=True)

    return entropy


def gaussian_kernel(x, bandwidth):
    batch_size, num_samples, dim = x.size()

    x_i = x.unsqueeze(2)  # (batch_size, num_samples, 1, dim)
    x_j = x.unsqueeze(1)  # (batch_size, 1, num_samples, dim)

    distances = torch.sum((x_i - x_j) ** 2, dim=-1)  # (batch_size, num_samples, num_samples)

    kernel_values = torch.exp(-distances / (2 * bandwidth**2))

    return kernel_values


class KDE:
    def __init__(self, kde_flag=True, marginal_flag=True):
        self.flag = kde_flag
        self.marginal_flag = marginal_flag

    def kde_entropy(self, x, k=1):
        batch_size, num_samples, dim = x.size()

        if self.flag:
            bandwidth = 1
            self.flag = False
        bandwidth = 1

        kernel_values = gaussian_kernel(x, bandwidth)  # (batch_size, num_samples, num_samples)

        density = kernel_values.sum(dim=2) / num_samples  # (batch_size, num_samples)
        _, indices = torch.topk(density, k=k, dim=1)
        sorted_indices = indices.squeeze(0).sort(dim=0)[0]
        x_max_likelihood = x[0, sorted_indices, :]

        log_density = torch.log(density + 1e-8)

        entropy = -log_density.mean(dim=1, keepdim=True)  # (batch_size, 1)

        return entropy

    def estimate_bandwidth(self, x, rule="scott"):
        num_samples, dim = x.size()

        std = x.std(dim=0).mean().item()
        if rule == "silverman":
            bandwidth = 1.06 * std * num_samples ** (-1 / 5)
        elif rule == "scott":
            bandwidth = std * num_samples ** (-1 / (dim + 4))
        else:
            raise ValueError("Unsupported rule. Choose 'silverman' or 'scott'.")

        return bandwidth
