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

"""Utilities for measuring alignment b/w language and vision model features.

Largely from the original PRH code:
https://github.com/minyoungg/platonic-rep/tree/main
"""

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from tqdm.auto import tqdm
from vprh import metrics


def prepare_features(feats, q=0.95, exact=False):
  """Prepare features by removing outliers and normalizing.

  Args:
      feats: a torch tensor of any share
      q: the quantile to remove outliers
      exact: Whether to compute the exact quantile.

  Returns:
      feats: a torch tensor of the same shape as the input
  """
  feats = metrics.remove_outliers(feats.float(), q=q, exact=exact)
  return feats.cuda()


def compute_score(
    x_feats,
    y_feats,
    metric='mutual_knn',
    topk=10,
    normalize=True,
):
  """Checks all layer combos of x_feats and y_feats for the best alignment.

  Args:
      x_feats: a torch tensor of shape N x L x D
      y_feats: a torch tensor of shape N x L x D
      metric: The alignment metric to use.
      topk: The number of nearest neighbors for k-NN metrics.
      normalize: Whether to normalize features before computing the score.

  Returns:
      best_alignment_score: the best alignment score
      best_alignment: the indices of the best alignment
  """
  # It's OK if L = 1, we'll repeat computation but it's minimal overhead for
  # clarity.
  if x_feats.ndim == 2:
    x_feats = x_feats.unsqueeze(1)

  if y_feats.ndim == 2:
    y_feats = y_feats.unsqueeze(1)

  best_alignment_indices = None
  all_scores = np.zeros((x_feats.shape[1] + 1, y_feats.shape[1] + 1))
  best_alignment_score = 0
  best_confusion = 0

  for i in range(-1, x_feats.shape[1]):
    x = x_feats.flatten(1, 2) if i == -1 else x_feats[:, i, :]

    for j in range(-1, y_feats.shape[1]):
      y = y_feats.flatten(1, 2) if j == -1 else y_feats[:, j, :]

      if normalize:
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

      score, neighbors = metrics.AlignmentMetrics.measure(
          metric, x, y, topk=topk
      )

      all_scores[i + 1, j + 1] = score
      if score > best_alignment_score:
        best_alignment_score = score
        best_alignment_indices = (i, j)
        best_confusion = neighbors

  feat_sizes = (x_feats.shape[1], y_feats.shape[1])
  return (
      best_alignment_score,
      best_alignment_indices,
      feat_sizes,
      best_confusion,
      all_scores,
  )


def compute_alignment(
    x_feat_paths,
    y_feat_paths,
    metric,
    topk,
    precise=True,
):
  """Computes alignment scores between two sets of features.

  Args:
      x_feat_paths: List of paths to x features.
      y_feat_paths: List of paths to y features.
      metric: The metric to use.
      topk: The number of nearest neighbors to use (specific to knn metrics).
      precise: If true use exact quantiling. (helpful to set to false if running
        on cpu) this is more of a feature to speed up matmul if using float32
        used in measure_alignment.py.

  Returns:
      A tuple containing alignment scores, indices, feature sizes, and
      nearest neighbor data.
  """
  symmetric_metric = x_feat_paths == y_feat_paths
  if metric == 'cycle_knn':
    symmetric_metric = False

  alignment_scores = np.zeros((len(x_feat_paths), len(y_feat_paths)))
  alignment_indices = np.zeros(
      (len(x_feat_paths), len(y_feat_paths), 2), dtype=int
  )
  total_feat_sizes = np.zeros(
      (len(x_feat_paths), len(y_feat_paths), 2), dtype=int
  )
  alignment_nn_data = np.empty(
      (len(x_feat_paths), len(y_feat_paths)), dtype=object
  )

  pbar = tqdm(total=len(y_feat_paths) * len(x_feat_paths))

  for i, x_fp in enumerate(x_feat_paths):
    print('Processing ', x_fp)
    feats = torch.load(x_fp, map_location='cuda:0', weights_only=True)
    x_feats = prepare_features(feats['feats'].float(), exact=precise)
    for j, y_fp in enumerate(y_feat_paths):
      pbar.update(1)

      if symmetric_metric:
        if i > j:
          pbar.update(1)
          continue
      feats = torch.load(y_fp, map_location='cuda:0', weights_only=False)
      all_feats = feats['feats'].float()
      y_feats = prepare_features(all_feats, exact=precise)

      best_score, best_indices, feat_sizes, best_confusion, full_scores = (
          compute_score(
              x_feats,
              y_feats,
              metric=metric,
              topk=topk,
          )
      )

      alignment_scores[i, j] = best_score
      alignment_indices[i, j] = best_indices
      total_feat_sizes[i, j] = feat_sizes
      alignment_nn_data[i, j] = best_confusion

      if symmetric_metric:
        alignment_scores[j, i] = best_score
        alignment_indices[j, i] = best_indices[::-1]

      del y_feats, full_scores
      torch.cuda.empty_cache()
    print('average score: ', np.mean(alignment_scores[i, :]))
    print()

  return (
      alignment_scores,
      alignment_indices,
      total_feat_sizes,
      alignment_nn_data,
  )
