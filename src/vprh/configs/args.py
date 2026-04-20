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

"""Script arguments for feature extraction and alignment."""

import dataclasses
from typing import Literal

from vprh.registry.llm import LLM_REGISTRY
from vprh.registry.llm import LLMConfig
from vprh.registry.video import VIDEO_MODEL_REGISTRY
from vprh.registry.video import VideoModelInterface


@dataclasses.dataclass
class ScriptArgs:
  """The complete set of arguments for configuring and running the script."""

  # --- Directory and Data Arguments ---
  feature_dir: str = './results/features'
  alignment_dir: str = './results/alignment'
  dataset: Literal['vatex', 'pvd'] = 'vatex'
  data_dir: str = 'data/vatex'
  annotation_path: str = 'vatex.data.img_png.pkl'

  # --- LLM Feature Extraction Arguments ---
  llm_pool: Literal['avg', 'last'] = 'avg'
  quantize: bool = False
  """Use quantization for LLMs (will have lower performance!)."""
  llm_caption_key: str = 'captions'
  """Key for the captions in the dataset."""
  num_captions: int = -1
  """Number of captions to use. -1 for all."""
  llm_names: list[str] = dataclasses.field(
      default_factory=lambda: list(LLM_REGISTRY.keys())
  )
  """List of LLM configurations to run. Defaults to all registered LLMs."""
  llm_only: bool = False
  """Process only LLMs, not video models."""

  # --- Video Model Feature Extraction Arguments ---
  video_pool: Literal['avg', 'cls'] = 'cls'
  video_sample_mode: Literal[
      'random', 'center', 'linspace', 'linspace_center'
  ] = 'center'
  num_frames: int = -1
  """Number of frames to sample. -1 for model default."""
  video_model_names: list[str] = dataclasses.field(
      default_factory=lambda: list(VIDEO_MODEL_REGISTRY.keys())
  )
  """List of video models to run. Defaults to all registered models."""
  video_only: bool = False
  """Process only video models, not LLMs."""

  # --- Processing and Execution Arguments ---
  batch_size: int = 2
  force_remake: bool = False
  force_download: bool = False

  # --- Alignment Arguments ---
  metric: Literal[
      'cycle_knn',
      'mutual_knn',
      'lcs_knn',
      'cka',
      'unbiased_cka',
      'cknna',
  ] = 'mutual_knn'
  topk: int = 10
  """Top-k for k-NN based metrics"""
  precise: bool = False
  """Use precise calculations (float32)"""
  use_short_name: bool = True
  """Use model short names for plotting."""

  # --- GPU Parallelization Arguments ---
  # These arguments are added to enable parallel execution.
  num_gpus: int = 1
  """Total number of GPUs for parallelization. (Set by launcher.py)"""
  gpu_id: int = 0
  """Worker ID for the current process, 0 to num_gpus-1. (Set by launcher.py)"""

  def __post_init__(self):
    """Called after the dataclass is initialized, instantiate configs."""
    assert not (
        self.llm_only and self.video_only
    ), "Can't specify both LLM & Video only"

    if self.llm_only:
      self.video_model_names = []
    self.video_models: list[VideoModelInterface] = []
    for model_name in self.video_model_names:
      if model_name not in VIDEO_MODEL_REGISTRY:
        raise ValueError(
            f'Unknown video model name: {model_name}.'
            f' Options: {VIDEO_MODEL_REGISTRY.keys()}'
        )
      self.video_models.append(VIDEO_MODEL_REGISTRY[model_name])

    if self.video_only:
      self.llm_names = []
    self.llm_configs: list[LLMConfig] = []
    for model_name in self.llm_names:
      if model_name not in LLM_REGISTRY:
        raise ValueError(
            f'Unknown LLM name: {model_name}. Options: {LLM_REGISTRY.keys()}'
        )
      self.llm_configs.append(LLM_REGISTRY[model_name])
