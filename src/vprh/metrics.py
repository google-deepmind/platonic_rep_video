# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Largely from the original PRH code.

https://github.com/minyoungg/platonic-rep/tree/main
"""

from typing import ClassVar

import numpy as np
import torch


try:
  import pymp

  pymp_available = True
except ImportError:
  pymp_available = False
  print(
      'Please install the pymp library using `pip install pymp` to speed up'
      'non-batched metrics'
  )


class AlignmentMetrics:
  """A collection of static methods for measuring alignment between features."""

  SUPPORTED_METRICS: ClassVar = [
      'cycle_knn',
      'mutual_knn',
      'lcs_knn',
      'cka',
      'unbiased_cka',
      'cknna',
  ]

  @staticmethod
  def measure(metric, *args, **kwargs):
    """Metric is a string for the function."""
    if metric not in AlignmentMetrics.SUPPORTED_METRICS:
      raise ValueError(f'Unrecognized metric: {metric}')

    return getattr(AlignmentMetrics, metric)(*args, **kwargs)

  @staticmethod
  def cycle_knn(feats_a, feats_b, topk):
    """LLM nearest neighbors -> Query Language Pair -> LVM nearest neighbors.

    Args:
        feats_a: A torch tensor of shape N x feat_dim.
        feats_b: A torch tensor of shape N x feat_dim.
        topk: The number of nearest neighbors to consider.

    Returns:
        acc: A float representing the accuracy.
    """
    _, knn_a = compute_nearest_neighbors(feats_a, topk)
    _, knn_b = compute_nearest_neighbors(feats_b, topk)
    return compute_knn_accuracy(knn_a[knn_b]).item()

  @staticmethod
  def mutual_knn(feats_a, feats_b, topk, return_values=False):
    """Computes the mutual KNN accuracy.

    Args:
        feats_a: A torch tensor of shape N x feat_dim.
        feats_b: A torch tensor of shape N x feat_dim.
        topk: The number of nearest neighbors to consider.
        return_values: Whether to return similarity scores and neighbors.

    Returns:
        A float representing the mutual KNN accuracy.
    """
    sim_a, knn_a = compute_nearest_neighbors(feats_a, topk)
    sim_b, knn_b = compute_nearest_neighbors(feats_b, topk)

    n = knn_a.shape[0]
    topk = knn_a.shape[1]

    # Create a range tensor for indexing
    range_tensor = torch.arange(n, device=knn_a.device).unsqueeze(1)

    # Create binary masks for knn_A and knn_B
    a_mask = torch.zeros(n, n, device=knn_a.device)
    b_mask = torch.zeros(n, n, device=knn_b.device)

    a_mask[range_tensor, knn_a] = 1.0
    b_mask[range_tensor, knn_b] = 1.0

    acc = (a_mask * b_mask).sum(dim=1) / topk

    if return_values:
      return (
          acc.mean().item(),
          torch.stack((knn_a, knn_b), dim=2).cpu().numpy(),
          torch.stack((sim_a, sim_b), dim=2).cpu().numpy(),
      )
    else:
      return (
          acc.mean().item(),
          torch.stack((knn_a, knn_b), dim=2).cpu().numpy(),
      )

  @staticmethod
  def lcs_knn(feats_a, feats_b, topk):
    """Computes the Longest Common Subsequence (LCS) score of KNNs."""
    _, knn_a = compute_nearest_neighbors(feats_a, topk)
    _, knn_b = compute_nearest_neighbors(feats_b, topk)
    score = longest_ordinal_sequence(knn_a, knn_b).float().mean()
    return score

  @staticmethod
  def cka(feats_a, feats_b, kernel_metric='ip', rbf_sigma=1.0, unbiased=False):
    """Computes the Centered Kernel Alignment (CKA) between features."""
    if kernel_metric == 'ip':
      # Compute kernel matrices for the linear case
      k = torch.mm(feats_a, feats_a.T)
      l_kernel = torch.mm(feats_b, feats_b.T)
    elif kernel_metric == 'rbf':
      # COMPUTES RBF KERNEL
      k = torch.exp(-(torch.cdist(feats_a, feats_a) ** 2) / (2 * rbf_sigma**2))
      l_kernel = torch.exp(
          -(torch.cdist(feats_b, feats_b) ** 2) / (2 * rbf_sigma**2)
      )
    else:
      raise ValueError(f'Invalid kernel metric {kernel_metric}')

    # Compute HSIC values
    hsic_fn = hsic_unbiased if unbiased else hsic_biased
    hsic_kk = hsic_fn(k, k)
    hsic_ll = hsic_fn(l_kernel, l_kernel)
    hsic_kl = hsic_fn(k, l_kernel)

    # Compute CKA
    # print('hsic', hsic_kl)
    cka_value = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)
    return cka_value.item()

  @staticmethod
  def unbiased_cka(*args, **kwargs):
    """A convenience method for calling CKA with the unbiased flag."""
    kwargs['unbiased'] = True
    return AlignmentMetrics.cka(*args, **kwargs)

  @staticmethod
  def cknna(
      feats_a, feats_b, topk=None, distance_agnostic=False, unbiased=True
  ):
    """Similarity-only CKA variant."""
    n = feats_a.shape[0]

    if topk < 2:
      raise ValueError('CKNNA requires topk >= 2')

    if topk is None:
      topk = feats_a.shape[0] - 1

    k = feats_a @ feats_a.T
    l_kernel = feats_b @ feats_b.T
    device = feats_a.device

    def similarity(k_mat, l_mat, topk):
      if unbiased:
        k_hat = k_mat.clone().fill_diagonal_(float('-inf'))
        l_hat = l_mat.clone().fill_diagonal_(float('-inf'))
      else:
        k_hat, l_hat = k_mat, l_mat

      # get topk indices for each row
      # if unbiased we cannot attend to the diagonal unless full topk
      # else we can attend to the diagonal
      _, topk_k_indices = torch.topk(k_hat, topk, dim=1)
      _, topk_l_indices = torch.topk(l_hat, topk, dim=1)

      # create masks for nearest neighbors
      mask_k = torch.zeros(n, n, device=device).scatter_(1, topk_k_indices, 1)
      mask_l = torch.zeros(n, n, device=device).scatter_(1, topk_l_indices, 1)

      # intersection of nearest neighbors
      mask = mask_k * mask_l

      if distance_agnostic:
        sim = mask * 1.0
      elif unbiased:
        sim = hsic_unbiased(mask * k_mat, mask * l_mat)
      else:
        sim = hsic_biased(mask * k_mat, mask * l_mat)
      return sim

    sim_kl = similarity(k, l_kernel, topk)
    sim_kk = similarity(k, k, topk)
    sim_ll = similarity(l_kernel, l_kernel, topk)
    return sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item()


def hsic_unbiased(k, l_kernel):
  """Compute the unbiased HSIC as per Equation 5 in the paper.

  > Reference: https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf
  """
  m = k.shape[0]

  # Zero out the diagonal elements of K and L
  k_tilde = k.clone().fill_diagonal_(0)
  l_tilde = l_kernel.clone().fill_diagonal_(0)

  # Compute HSIC using the formula in Equation 5
  hsic_value = (
      (torch.sum(k_tilde * l_tilde.T))
      + (torch.sum(k_tilde) * torch.sum(l_tilde) / ((m - 1) * (m - 2)))
      - (2 * torch.sum(torch.mm(k_tilde, l_tilde)) / (m - 2))
  )

  hsic_value /= m * (m - 3)
  return hsic_value


def hsic_biased(k, l_kernel):
  """Compute the biased HSIC (the original CKA)."""
  h = torch.eye(k.shape[0], dtype=k.dtype, device=k.device) - 1 / k.shape[0]
  return torch.trace(k @ h @ l_kernel @ h)


def compute_knn_accuracy(knn):
  """Compute the accuracy of the nearest neighbors..

  Assumes index is the gt label.

  Args:
      knn: a torch tensor of shape N x topk

  Returns:
      acc: a float representing the accuracy
  """
  n = knn.shape[0]
  acc = knn == torch.arange(n, device=knn.device).view(-1, 1, 1)
  acc = acc.float().view(n, -1).max(dim=1).values.mean()
  return acc


def compute_nearest_neighbors(feats, topk=1):
  """Compute the nearest neighbors of feats.

  Args:
      feats: a torch tensor of shape N x D
      topk: the number of nearest neighbors to return

  Returns:
      knn: a torch tensor of shape N x topk, or a tuple of (sim, knn) if
      return_values is True.
  """
  assert feats.ndim == 2, f'Expected feats to be 2D, got {feats.ndim}'
  sim, knn = (feats @ feats.T).fill_diagonal_(-1e8).sort(dim=1, descending=True)
  sim = sim[:, :topk]
  knn = knn[:, :topk]
  return sim, knn


def longest_ordinal_sequence(x_seq, y_seq):
  """For each pair in x_seq and y_seq, compute length of the LCS."""

  def lcs_length(x, y):
    """Compute the length of the longest common subsequence between (x, y).

    This is a classic dynamic programming implementation.
    """
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
      for j in range(1, n + 1):
        if x[i - 1] == y[j - 1]:
          dp[i][j] = dp[i - 1][j - 1] + 1
        else:
          dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

  lcs = compute_distance(x_seq, y_seq, lcs_length)
  return lcs


def compute_distance(x_in, y_in, dist_fn):
  """Compute distance in parallel."""
  b, _ = x_in.shape
  distances = np.zeros(b)
  x_in, y_in = x_in.cpu().numpy(), y_in.cpu().numpy()

  if pymp_available:
    with pymp.Parallel(4) as p:
      for i in p.range(b):
        distances[i] = dist_fn(x_in[i], y_in[i])
  else:
    for i in range(b):
      distances[i] = dist_fn(x_in[i], y_in[i])
  return torch.tensor(distances)


def remove_outliers(feats, q, exact=False, max_threshold=None):
  """Removes outliers from a tensor by clamping values.

  Args:
      feats: The input tensor.
      q: The quantile to use for determining the threshold.
      exact: Whether to compute the exact quantile.
      max_threshold: An optional maximum threshold value.
  """
  if q == 1:
    return feats

  if exact:
    # sorts the whole tensor and gets the q-th percentile
    q_val = feats.view(-1).abs().sort().values[int(q * feats.numel())]
  else:
    # quantile for element in the tensor and take the average
    q_val = torch.quantile(feats.abs().flatten(start_dim=1), q, dim=1).mean()

  if max_threshold is not None:
    max_threshold = max(max_threshold, q_val)

  return feats.clamp(-q_val, q_val)
