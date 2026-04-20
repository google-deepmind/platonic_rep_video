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

"""Plots alignment as a matrix of LLM models vs Vision models."""

import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from registry import CONFIGS
import tyro
from vprh.configs.args import ScriptArgs


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
  """Truncates a colormap to a specific range."""
  new_cmap = mcolors.LinearSegmentedColormap.from_list(
      f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
      cmap(np.linspace(minval, maxval, n)),
  )
  return new_cmap


def plot_matrix(fp: Path):
  """Plots a matrix from a numpy array file.

  Features like global + row averages, and customizable file path.

  Args:
    fp: File path to the numpy array file.
  """
  result = np.load(fp, allow_pickle=True).item()
  a = result['scores']
  b = result['indices']
  c = result['feat_sizes']
  x_labels = result['llm_models']
  y_labels = result['vision_models']
  b = np.concatenate((b, c), axis=2)

  col_avg = np.mean(a, axis=0)
  sort_indices = np.argsort(col_avg)
  a = a[:, sort_indices]
  b = b[:, sort_indices, :]
  y_labels = [y_labels[i] for i in sort_indices]

  row_avg = np.mean(a, axis=1)
  row_sort_indices = np.argsort(row_avg)[::-1]
  a = a[row_sort_indices, :]
  b = b[row_sort_indices, :, :]
  x_labels = [x_labels[i] for i in row_sort_indices]

  a = np.swapaxes(a, 0, 1)
  b = np.swapaxes(b, 0, 1)

  global_avg = np.mean(a)
  global_max = np.max(a)

  norm = mcolors.Normalize(vmin=0.15, vmax=global_max)
  cmap = plt.get_cmap('hot')
  trunc_cmap = truncate_colormap(cmap, 0.25, 1.0)

  plt.rcParams.update({'font.size': 14})
  plt.figure(figsize=(22, 10))
  plt.imshow(a, cmap=trunc_cmap, interpolation='nearest', norm=norm)

  plt.title(f'Global Average: {global_avg:.4f}', fontsize=16)

  for i, row in enumerate(a):
    row_avg_val = np.max(row)
    y_labels[i] += f' (Max: {row_avg_val:.4f})'
    y_labels[i] += f' l:{int(b[i][0][2])} '

  for j in range(a.shape[1]):
    col = a[:, j]
    col_avg_val = np.mean(col)
    x_labels[j] += f' (Avg: {col_avg_val:.3f})'
    x_labels[j] += f' l:{int(b[0][j][3])} '

  plt.xticks(
      range(len(x_labels)), x_labels, rotation=20, ha='right', fontsize=12
  )
  plt.yticks(range(len(y_labels)), y_labels, fontsize=12)

  for i in range(b.shape[0]):
    for j in range(b.shape[1]):
      plt.text(
          j,
          i,
          f'{a[i, j]:.3f}\n({int(b[i, j][0])},{int(b[i, j][1])})',
          ha='center',
          va='center',
          color='black',
          fontsize=10,
      )

  plt.tight_layout()
  output_filename = fp.with_suffix('.png')
  print(f'Saving plot to {output_filename}')
  plt.savefig(output_filename, bbox_inches='tight')


def main(args: ScriptArgs):
  """Main function to plot the alignment matrix."""
  input_dir_name = Path(args.feature_dir).name
  save_dir = Path(args.alignment_dir) / input_dir_name / args.dataset
  metric_suffix = f'_k{args.topk}' if 'knn' in args.metric else ''
  load_path = save_dir / f'{args.metric}{metric_suffix}.npy'

  if not os.path.exists(load_path):
    print(f'Alignment file not found at {load_path}')
    print('Please run measure_alignment.py first.')
    return

  plot_matrix(load_path)


if __name__ == '__main__':
  args: ScriptArgs = tyro.extras.overridable_config_cli(CONFIGS)
  main(args)
