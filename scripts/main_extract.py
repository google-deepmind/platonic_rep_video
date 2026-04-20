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

"""Main script for feature extraction."""

from registry import CONFIGS
import tyro
from vprh import dataloaders as data
from vprh import extract_features
from vprh.configs.args import ScriptArgs


def main():
  """Main function to run feature extraction."""
  args: ScriptArgs = tyro.extras.overridable_config_cli(CONFIGS)

  if args.quantize:
    print('quantize is set to True. The alignment score will be slightly off.')

  # --- Workload Distribution ---
  # Select a subset of models for this worker based on its assigned gpu_id.
  my_llm_configs = args.llm_configs[args.gpu_id :: args.num_gpus]
  my_video_models = args.video_models[args.gpu_id :: args.num_gpus]

  if args.gpu_id == 0:  # Print dataset info only from the first worker
    print(f'Distributing work over {args.num_gpus} GPUs.')
  print(
      f'[GPU-{args.gpu_id}] processing {len(my_llm_configs)} LLMs and'
      f' {len(my_video_models)} video models.'
  )

  if my_llm_configs:
    print(
        f'Processing Language Features on {args.dataset} w/'
        f' {args.num_captions} captions...'
    )
    if args.dataset == 'vatex':
      dataset = data.create_vatex_captions(args.annotation_path)
    elif args.dataset == 'pvd':
      dataset = data.create_pvd_captions(args.annotation_path)
    else:
      raise ValueError(f'Unknown dataset {args.dataset}')

    for llm_config in my_llm_configs:
      extract_features.extract_llm_features(llm_config, dataset, args)

  if my_video_models:
    print(
        f'Processing Video Features on {args.dataset} w/'
        f' {args.num_frames} frames..'
    )
    if args.dataset == 'vatex':
      dataset = data.VaTeX(
          args.annotation_path, process_mode=args.video_sample_mode
      )
    elif args.dataset == 'pvd':
      dataset = data.PerceptionVideoDataset(
          args.annotation_path,
          args.data_dir,
          process_mode=args.video_sample_mode,
      )
    else:
      raise ValueError(f'Unknown dataset {args.dataset}')

    for video_model in my_video_models:
      extract_features.extract_video_features(video_model, dataset, args)


if __name__ == '__main__':
  main()
