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

"""Measures alignment between language and vision model features."""

import os
from pathlib import Path
from pprint import pprint
import sys

import numpy as np
from registry import CONFIGS
import torch
import tyro
from vprh.alignment import compute_alignment
from vprh.configs.args import ScriptArgs
from vprh.registry.llm import LLM_REGISTRY
from vprh.registry.video import VIDEO_MODEL_REGISTRY


def main(args: ScriptArgs):
  """Main function to run the alignment measurement."""
  if not args.precise:
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

  input_dir = Path(args.feature_dir) / args.dataset
  input_dir_name = Path(args.feature_dir).name
  save_dir = Path(args.alignment_dir) / input_dir_name / args.dataset
  os.makedirs(save_dir, exist_ok=True)

  metric_suffix = f'_k{args.topk}' if 'knn' in args.metric else ''
  save_path = save_dir / f'{args.metric}{metric_suffix}.npy'

  print('saving to', save_path)

  if os.path.exists(save_path) and not args.force_remake:
    print(f'alignment already exists at {save_path}')
    sys.exit()
  elif os.path.exists(save_path) and args.force_remake:
    print(f'remaking alignment at {save_path}')

  llm_model_paths = []
  video_model_paths = []
  llm_models = []
  video_models = []

  print(f'Searching for feature files in: {input_dir}')
  for model_feat in sorted(Path(input_dir).glob('*.pt')):
    name = model_feat.stem.split('_pool')[0]
    key_name = name if args.use_short_name else model_feat.stem
    if name in LLM_REGISTRY:
      llm_models.append(key_name)
      llm_model_paths.append(str(model_feat))
    elif name in VIDEO_MODEL_REGISTRY:
      video_models.append(key_name)
      video_model_paths.append(str(model_feat))

  print(
      f'Found {len(llm_models)} LLM features and {len(video_models)} vision'
      ' features.'
  )
  pprint(llm_models)
  pprint(video_models)
  if not llm_model_paths or not video_model_paths:
    print(
        'Not enough features to compute alignment. Need at least one LLM and'
        ' one vision feature file.'
    )
    sys.exit()

  print('\nMeasuring alignment...')
  print(f'Metric: {args.metric}, Top-k: {args.topk}')

  alignment_scores, alignment_indices, total_feat_sizes, alignment_nn_data = (
      compute_alignment(
          llm_model_paths,
          video_model_paths,
          args.metric,
          args.topk,
          args.precise,
      )
  )

  payload = {
      'scores': alignment_scores,
      'indices': alignment_indices,
      'feat_sizes': total_feat_sizes,
      'nn_data': alignment_nn_data,
      'llm_models': llm_models,
      'vision_models': video_models,
      'llm_paths': llm_model_paths,
      'vision_paths': video_model_paths,
  }

  np.save(save_path, payload)
  print(f'Saved alignment results to {save_path}')

  # Find the best alignment score and its corresponding models
  best_score_idx = np.unravel_index(
      np.argmax(alignment_scores), alignment_scores.shape
  )
  best_llm_model = llm_models[best_score_idx[0]]
  best_video_model = video_models[best_score_idx[1]]
  best_alignment_score = alignment_scores[best_score_idx]
  best_alignment_indices = alignment_indices[best_score_idx]

  # LLM layers sometimes greater by 1 arbitrarily for token embeddings
  print(f'\nBest Alignment Score: {best_alignment_score:.4f}')
  print(
      f'  LLM Model: {best_llm_model} (Layer {int(best_alignment_indices[0])}'
      f' of {total_feat_sizes[best_score_idx][0]})'
  )
  print(
      f'  Vision Model: {best_video_model} (Layer'
      f' {int(best_alignment_indices[1])} of'
      f' {total_feat_sizes[best_score_idx][1]})'
  )


if __name__ == '__main__':
  args: ScriptArgs = tyro.extras.overridable_config_cli(CONFIGS)
  main(args)
