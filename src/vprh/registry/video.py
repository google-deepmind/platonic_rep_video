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

"""Video model registry with models from huggingface and timm."""

import abc
from dataclasses import dataclass
from dataclasses import field
import gc
import types
from typing import Any

from einops import rearrange
import numpy as np
import timm
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
from transformers import AutoConfig
from transformers import AutoImageProcessor
from transformers import AutoModel
from transformers import AutoVideoProcessor
from transformers import Dinov2Model
from transformers import TimesformerModel
from transformers import VideoMAEImageProcessor
from transformers import VivitImageProcessor
from transformers import VivitModel


# ruff: noqa: D101, D102, D107


VIDEO_MODEL_REGISTRY = {}


def register(name: str):
  """Decorator to register a video model class in the registry."""

  def decorator(cls):
    if name in VIDEO_MODEL_REGISTRY:
      raise ValueError(f"Model '{name}' is already registered.")
    VIDEO_MODEL_REGISTRY[name] = cls(short_name=name)
    return cls

  return decorator


@dataclass
class VideoModelConfig:
  """Configuration for video models."""

  model_name: str
  short_name: str | None = None
  image_processor_name: str | None = None

  clip_len: int = 16
  """Number of frames to sample from each video."""
  frame_sample_rate: int = 4
  """The rate at which to sample frames from the video."""
  native_num_frames: int = 16
  """The number of frames the model was trained on and expects as input.
        This may differ from `clip_len`, esp for image models."""
  pool_types: list[str] = field(default_factory=lambda: ['cls', 'avg'])
  """Pool types supported by each model. Ideally we use cls tokens
        when available, but not every model has them."""

  def __post_init__(self):
    """Automatically fill in image_processor_name with model name."""
    if self.image_processor_name is None:
      self.image_processor_name = self.model_name


class VideoModelInterface(abc.ABC):
  """Abstract base class defining the interface for a model in the registry."""

  def __init__(self, config: VideoModelConfig):
    """Lazy init the models."""
    self.config: VideoModelConfig = config
    self.device = torch.device('cpu')
    self.model: torch.nn.Module | None = None
    self._raw_processor = None
    self._initialized = False

  @abc.abstractmethod
  def _load_model(self) -> Any:
    """Loads the model, returns the model object."""

  @abc.abstractmethod
  def _load_raw_processor(self) -> Any:
    """Loads the preprocessor, returns the processor object."""

  def initialize(self) -> 'VideoModelInterface':
    """Loads model and processor into memory."""
    if not self._initialized:
      print(f'Initializing {self.config.short_name}...')
      self.model = self._load_model()
      self._raw_processor = self._load_raw_processor()
      if self.model:
        self.model.to(self.device)
      self._initialized = True
    return self

  def cleanup(self) -> None:
    """Removes model and processor from memory."""
    if self._initialized:
      del self.model
      del self._raw_processor

      self.model = None
      self._raw_processor = None
      self._initialized = False
      gc.collect()
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
      print(f'{self.config.short_name} cleaned up.')
    else:
      print(f'{self.config.short_name} was not initialized, no cleanup needed.')

  def _ensure_initialized(self) -> None:
    if not self._initialized:
      raise RuntimeError(
          f'{self.config.short_name} is not initialized. '
          'Call .initialize() or .to(device) before using this method.'
      )

  def to(self, device: str | torch.device) -> 'VideoModelInterface':
    """Moves the model to the specified device, initializing if necessary."""
    self.device = torch.device(device)
    if not self._initialized:
      print(
          f'{self.config.short_name} not initialized. Initializing and moving'
          f' to {self.device}.'
      )
      self.initialize()  # initialize() will move to self.device
    elif self.model:
      self.model.to(self.device)
      print(f'{self.config.short_name} moved to {self.device}.')
    return self

  @abc.abstractmethod
  def preprocess(self, images: list[np.ndarray]) -> torch.Tensor:
    """Processes a list of video frames into a tensor for the model.

    Args:
        images: A list of NumPy arrays, where each array represents a single
          frame of shape (H, W, C).

    Returns:
        A torch.Tensor ready for the model, typically of shape
        (B, F, C, H, W) or (B, C, F, H, W), where F is the number of frames.
    """

  @abc.abstractmethod
  def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    """Performs a forward pass to get the final feature representation.

    Args:
        pixel_values: The preprocessed video tensor.

    Returns:
        A tensor representing the final output of the model,
        typically of shape (B, C).
    """

  @abc.abstractmethod
  def forward_intermediates(
      self, pixel_values: torch.Tensor, pool_type: str = 'cls'
  ) -> torch.Tensor:
    """Performs a forward pass to extract intermediate layer features.

    Args:
        pixel_values: The preprocessed video tensor.
        pool_type: The pooling strategy to apply (e.g., 'cls', 'avg').

    Returns:
        A tensor of intermediate layer features, typically of shape (B, L, C),
        where L is the number of layers.
    """

  @torch.no_grad()
  def __call__(
      self, pixel_values: torch.Tensor, pool_type: str = 'cls'
  ) -> torch.Tensor:
    """Alias for forward_intermediates."""
    assert (
        pool_type in self.config.pool_types
    ), f'Only supported pool types: {self.config.pool_types}'
    if not self._initialized:
      self.initialize()
    return self.forward_intermediates(pixel_values, pool_type)


class HuggingFaceModelBase(VideoModelInterface):

  def _load_model(self) -> Any:
    hf_config = AutoConfig.from_pretrained(
        self.config.model_name, trust_remote_code=True
    )
    return AutoModel.from_pretrained(
        self.config.model_name, config=hf_config, trust_remote_code=True
    )

  def _load_raw_processor(self) -> Any:
    return AutoImageProcessor.from_pretrained(
        self.config.image_processor_name, trust_remote_code=True
    )

  def preprocess(self, images: list[np.ndarray]) -> torch.Tensor:
    self._ensure_initialized()

    inputs = self._raw_processor(images=images, return_tensors='pt')
    return inputs['pixel_values']

  def forward(
      self, pixel_values: torch.Tensor, pool_type: str = 'cls'
  ) -> torch.Tensor:
    """Return [B, C] output."""
    self._ensure_initialized()
    pixel_values = pixel_values.to(self.device)
    outputs = self.model(pixel_values=pixel_values)
    if pool_type == 'cls':
      return outputs.last_hidden_state[:, 0, :]
    elif pool_type == 'avg':
      return outputs.last_hidden_state[:, 1:, :].mean(dim=1)
    else:
      raise ValueError(f'Invalid pool_type: {pool_type}')

  def forward_intermediates(
      self, pixel_values: torch.Tensor, pool_type: str = 'cls'
  ) -> torch.Tensor:
    """Returns [B, L, C] output, L = number of layers."""
    assert (
        pool_type in self.config.pool_types
    ), f'Only supported pool types: {self.config.pool_types}'
    self._ensure_initialized()
    pixel_values = pixel_values.to(self.device)
    # Skipping initial embedding to only extract intermediate features
    hidden_states = outputs.hidden_states[1:]
    if pool_type == 'cls':
      feats = [v[:, 0, :] for v in hidden_states]
    elif pool_type == 'avg':
      feats = [v[:, 1:, :].mean(dim=1) for v in hidden_states]
    all_layer_features = torch.stack(feats, dim=1)
    return all_layer_features


class TimmModelBase(VideoModelInterface):

  def _load_model(self) -> Any:
    return timm.create_model(
        self.config.model_name, pretrained=True, num_classes=0
    )

  def _load_raw_processor(self) -> Any:
    # Assume that model is always loaded before processor
    # since _load_model is called inside initialize()
    processor = create_transform(
        **resolve_data_config(self.model.pretrained_cfg, model=self.model)
    )
    return processor

  def preprocess(self, frames: list[np.ndarray]) -> torch.Tensor:
    self._ensure_initialized()
    return self._raw_processor(frames)

  def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    self._ensure_initialized()
    pixel_values = pixel_values.to(self.device)
    features = self.model.forward_features(pixel_values)
    return torch.mean(features, dim=1)

  def forward_intermediates(
      self, pixel_values: torch.Tensor, pool_type: str = 'cls'
  ) -> torch.Tensor:
    assert (
        pool_type in self.config.pool_types
    ), f'Only supported pool types: {self.config.pool_types}'
    self._ensure_initialized()
    pixel_values = pixel_values.to(self.device)
    hidden_states = self.model.forward_intermediates(pixel_values)
    if pool_type == 'cls':
      feats = [v[:, 0, :] for v in hidden_states]
    elif pool_type == 'avg':
      feats = [v[:, 1:, :].mean(dim=1) for v in hidden_states]
    all_layer_features = torch.stack(feats, dim=1)
    return all_layer_features


class TimmImageAsVideoModelBase(TimmModelBase):
  """Does reshaping to collapse batch and frames together before processing."""

  def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    self._ensure_initialized()
    pixel_values = pixel_values.to(self.device)
    b, f, c, h, w = pixel_values.shape
    pixel_values_reshaped = pixel_values.view(b * f, c, h, w)
    features = self.model.forward_features(pixel_values_reshaped)
    cls_tokens = features[:, 0, :].view(b, f, -1)
    return torch.mean(cls_tokens, dim=1)

  def forward_intermediates(
      self, pixel_values: torch.Tensor, pool_type: str = 'cls'
  ) -> torch.Tensor:
    assert (
        pool_type in self.config.pool_types
    ), f'Only supported pool types: {self.config.pool_types}'
    self._ensure_initialized()
    pixel_values = pixel_values.to(self.device)
    b, f, c, h, w = pixel_values.shape
    pixel_values_reshaped = pixel_values.view(b * f, c, h, w)
    hidden_states = self.model.forward_intermediates(
        pixel_values_reshaped,
        output_fmt='NLC',
        intermediates_only=True,
        return_prefix_tokens=pool_type == 'cls',
        norm=True,
    )
    if pool_type == 'cls':
      # Second arg in tuple is the class token
      feats = [v[1].squeeze(dim=1) for v in hidden_states]
    elif pool_type == 'avg':
      feats = [v.mean(dim=1) for v in hidden_states]
    else:
      raise ValueError(f'Invalid pool_type: {pool_type}')
    all_layer_features = torch.stack(feats, dim=1)
    all_layer_features = rearrange(
        all_layer_features, '(B F) L C -> B F L C', B=b
    )
    # By default average over all frames
    all_layer_features = all_layer_features.mean(dim=1)
    return all_layer_features


# --- Model Registrations ---
@register('timesformer_base_k400')
class TimeSFormerBaseK400(HuggingFaceModelBase):
  def __init__(self, short_name: str | None = None):
    super().__init__(
        VideoModelConfig(
            model_name='facebook/timesformer-base-finetuned-k400',
            short_name=short_name,
            image_processor_name='MCG-NJU/videomae-base',
            clip_len=8,
            native_num_frames=8,
        )
    )

  def _load_model(self):
    return TimesformerModel.from_pretrained(self.config.model_name)


@register('timesformer_base_ssv2')
class TimeSFormerBaseSSv2(HuggingFaceModelBase):

  def __init__(self, short_name: str | None = None):
    super().__init__(
        VideoModelConfig(
            model_name='facebook/timesformer-base-finetuned-ssv2',
            short_name=short_name,
            clip_len=8,
            native_num_frames=8,
        )
    )


@register('videomaev2_base')
class VideoMAEv2Base(HuggingFaceModelBase):

  def __init__(
      self, config: VideoModelConfig = None, short_name: str | None = None
  ):
    # Support inheritance of the short_names this way
    super().__init__(
        config
        if config is not None
        else VideoModelConfig(
            model_name='OpenGVLab/VideoMAEv2-Base',
            short_name=short_name,
            frame_sample_rate=2,
            pool_types=['avg'],
        )
    )

  def _load_raw_processor(self) -> Any:
    return VideoMAEImageProcessor.from_pretrained(
        self.config.image_processor_name, trust_remote_code=True
    )

  def preprocess(self, images: list[np.ndarray]) -> torch.Tensor:
    """Override to channels_last for timm style model."""
    self._ensure_initialized()

    inputs = self._raw_processor(images=images, return_tensors='pt')
    return inputs['pixel_values']

  def _load_model(self) -> Any:
    """Monkey patch the model load to support intermediate layers."""
    hf_config = AutoConfig.from_pretrained(
        self.config.model_name, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        self.config.model_name, config=hf_config, trust_remote_code=True
    )

    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: list[int] | None = None,
        norm: bool = True,
    ):
      intermediates = []
      take_indices = []
      if indices is None:
        take_indices = list(range(len(self.blocks)))

      b = x.size(0)

      x = self.patch_embed(x)

      if self.pos_embed is not None:
        x = (
            x
            + self.pos_embed.expand(b, -1, -1)
            .type_as(x)
            .to(x.device)
            .clone()
            .detach()
        )
      x = self.pos_drop(x)

      for i, blk in enumerate(self.blocks):
        x = blk(x)
        if i in take_indices:
          intermediates.append(self.fc_norm(x.mean(1)) if norm else x.mean(1))

      return intermediates

    model.model.forward_intermediates = types.MethodType(
        forward_intermediates, model.model
    )
    return model

  def forward(
      self, pixel_values: torch.Tensor, pool_type: str = 'avg'
  ) -> torch.Tensor:
    self._ensure_initialized()
    pixel_values = pixel_values.to(self.device)
    features = self.model.forward_features(pixel_values)
    return torch.mean(features, dim=1)

  def forward_intermediates(
      self, pixel_values: torch.Tensor, pool_type: str = 'avg'
  ):
    """Only pooling supported."""
    self._ensure_initialized()
    # [B, T, C, H, W] -> [B, C, T, H, W]
    pixel_values = pixel_values.permute(0, 2, 1, 3, 4).to(self.device)
    hidden_states = self.model.model.forward_intermediates(pixel_values)
    all_layer_features = torch.stack(hidden_states, dim=1)
    return all_layer_features


@register('videomaev2_large')
class VideoMAEv2Large(VideoMAEv2Base):

  def __init__(self, short_name: str | None = None):
    config = VideoModelConfig(
        model_name='OpenGVLab/VideoMAEv2-Large',
        short_name=short_name,
        frame_sample_rate=2,
    )
    super().__init__(config)


@register('videomaev2_huge')
class VideoMAEv2Huge(VideoMAEv2Base):

  def __init__(self, short_name: str | None = None):
    config = VideoModelConfig(
        model_name='OpenGVLab/VideoMAEv2-Huge',
        short_name=short_name,
        frame_sample_rate=2,
    )
    super().__init__(config)


@register('videomae_base_ssv2')
class VideoMAEBaseSSv2(HuggingFaceModelBase):

  def __init__(self, short_name: str | None = None):
    super().__init__(
        VideoModelConfig(
            model_name='MCG-NJU/videomae-base-ssv2',
            short_name=short_name,
            frame_sample_rate=2,
        )
    )


@register('videomae_base_finetuned_kinetics')
class VideoMAEBaseFinetunedKinetics(HuggingFaceModelBase):

  def __init__(self, short_name: str | None = None):
    super().__init__(
        VideoModelConfig(
            model_name='MCG-NJU/videomae-base-finetuned-kinetics',
            short_name=short_name,
            frame_sample_rate=2,
        )
    )


@register('videomae_huge_finetuned_kinetics')
class VideoMAEHugeFinetunedKinetics(HuggingFaceModelBase):

  def __init__(self, short_name: str | None = None):
    super().__init__(
        VideoModelConfig(
            model_name='MCG-NJU/videomae-huge-finetuned-kinetics',
            short_name=short_name,
            frame_sample_rate=1,
        )
    )


@register('vivit_b_16x2_kinetics400')
class ViViTBase16x2Kinetics400(HuggingFaceModelBase):

  def __init__(self, short_name: str | None = None):
    super().__init__(
        VideoModelConfig(
            model_name='google/vivit-b-16x2-kinetics400',
            short_name=short_name,
            clip_len=32,
            frame_sample_rate=1,
            native_num_frames=32,
        )
    )

  def _load_model(self):
    return VivitModel.from_pretrained(self.config.model_name)

  def _load_raw_processor(self):
    return VivitImageProcessor.from_pretrained(self.config.model_name)


# --- TIMM Image Models as Video ---
@register('dinov2_small_video')
class DINOv2SmallPatch14Video(TimmImageAsVideoModelBase):

  def __init__(
      self, config: VideoModelConfig = None, short_name: str | None = None
  ):
    super().__init__(
        config
        if config is not None
        else VideoModelConfig(
            model_name='vit_small_patch14_dinov2.lvd142m',
            short_name=short_name,
            clip_len=8,
            frame_sample_rate=4,
            native_num_frames=1,
        )
    )

  def preprocess(self, frames: list[np.ndarray]) -> torch.Tensor:
    self._ensure_initialized()
    return self._raw_processor(frames)


@register('dinov2_small_image')
class DINOv2SmallPatch14Image(DINOv2SmallPatch14Video):

  def __init__(self, short_name: str | None = None):
    config = DINOv2SmallPatch14Video(short_name=short_name).config
    config.clip_len = 1
    config.frame_sample_rate = 1
    super().__init__(config=config)


@register('dinov2_base_video')
class DINOv2BasePatch14Video(DINOv2SmallPatch14Video):

  def __init__(self, short_name: str | None = None):
    config = DINOv2SmallPatch14Video(short_name=short_name).config
    config.model_name = 'vit_base_patch14_dinov2.lvd142m'
    super().__init__(config=config)


@register('dinov2_base_image')
class DINOv2BasePatch14Image(DINOv2SmallPatch14Video):

  def __init__(self, short_name: str | None = None):
    config = DINOv2SmallPatch14Image(short_name=short_name).config
    config.model_name = 'vit_base_patch14_dinov2.lvd142m'
    super().__init__(config=config)


@register('dinov2_large_video')
class DINOv2LargePatch14Video(DINOv2SmallPatch14Video):

  def __init__(self, short_name: str | None = None):
    config = DINOv2SmallPatch14Video(short_name=short_name).config
    config.model_name = 'vit_large_patch14_dinov2.lvd142m'
    super().__init__(config=config)


@register('dinov2_large_image')
class DINOv2LargePatch14Image(DINOv2SmallPatch14Video):

  def __init__(self, short_name: str | None = None):
    config = DINOv2SmallPatch14Image(short_name=short_name).config
    config.model_name = 'vit_large_patch14_dinov2.lvd142m'
    super().__init__(config=config)


@register('dinov2_giant_video')
class DINOv2GiantPatch14Video(DINOv2SmallPatch14Video):

  def __init__(self, short_name: str | None = None):
    config = DINOv2SmallPatch14Video(short_name=short_name).config
    config.model_name = 'vit_giant_patch14_dinov2.lvd142m'
    super().__init__(config=config)


@register('dinov2_giant_image')
class DINOv2GiantPatch14Image(DINOv2SmallPatch14Video):

  def __init__(self, short_name: str | None = None):
    config = DINOv2SmallPatch14Image(short_name=short_name).config
    config.model_name = 'vit_giant_patch14_dinov2.lvd142m'
    super().__init__(config=config)


# --- Other TIMM Image Models as Video ---
@register('augreg_tiny_video')
class AugRegTinyVideo(TimmImageAsVideoModelBase):

  def __init__(
      self, config: VideoModelConfig = None, short_name: str | None = None
  ):
    super().__init__(
        config
        if config is not None
        else VideoModelConfig(
            model_name='vit_tiny_patch16_224.augreg_in21k',
            short_name=short_name,
            clip_len=8,
            frame_sample_rate=4,
            native_num_frames=1,
        )
    )


@register('augreg_tiny_image')
class AugRegTinyImage(AugRegTinyVideo):

  def __init__(self, short_name: str | None = None):
    config = AugRegTinyVideo(short_name=short_name).config
    config.clip_len = 1
    config.frame_sample_rate = 1
    super().__init__(config=config)


@register('augreg_small_video')
class AugRegSmallVideo(AugRegTinyVideo):

  def __init__(self, short_name: str | None = None):
    config = AugRegTinyVideo(short_name=short_name).config
    config.model_name = 'vit_small_patch16_224.augreg_in21k'
    super().__init__(config=config)


@register('augreg_small_image')
class AugRegSmallImage(AugRegTinyVideo):

  def __init__(self, short_name: str | None = None):
    config = AugRegTinyImage(short_name=short_name).config
    config.model_name = 'vit_small_patch16_224.augreg_in21k'
    super().__init__(config=config)


@register('augreg_base_video')
class AugRegBaseVideo(AugRegTinyVideo):

  def __init__(self, short_name: str | None = None):
    config = AugRegTinyVideo(short_name=short_name).config
    config.model_name = 'vit_base_patch16_224.augreg_in21k'
    super().__init__(config=config)


@register('augreg_base_image')
class AugRegBaseImage(AugRegTinyVideo):

  def __init__(self, short_name: str | None = None):
    config = AugRegTinyImage(short_name=short_name).config
    config.model_name = 'vit_base_patch16_224.augreg_in21k'
    super().__init__(config=config)


@register('augreg_large_video')
class AugRegLargeVideo(AugRegTinyVideo):

  def __init__(self, short_name: str | None = None):
    config = AugRegTinyVideo(short_name=short_name).config
    config.model_name = 'vit_large_patch16_224.augreg_in21k'
    super().__init__(config=config)


@register('augreg_large_image')
class AugRegLargeImage(AugRegTinyVideo):

  def __init__(self, short_name: str | None = None):
    config = AugRegTinyImage(short_name=short_name).config
    config.model_name = 'vit_large_patch16_224.augreg_in21k'
    super().__init__(config=config)


@register('mae_base_video')
class MAEBaseVideo(TimmImageAsVideoModelBase):

  def __init__(
      self, config: VideoModelConfig = None, short_name: str | None = None
  ):
    super().__init__(
        config
        if config is not None
        else VideoModelConfig(
            model_name='vit_base_patch16_224.mae',
            short_name=short_name,
            clip_len=8,
            frame_sample_rate=4,
            native_num_frames=1,
        )
    )


@register('mae_base_image')
class MAEBaseImage(MAEBaseVideo):

  def __init__(self, short_name: str | None = None):
    config = MAEBaseVideo(short_name=short_name).config
    config.clip_len = 1
    config.frame_sample_rate = 1
    super().__init__(config=config)


@register('mae_large_video')
class MAELargeVideo(MAEBaseVideo):

  def __init__(self, short_name: str | None = None):
    config = MAEBaseVideo(short_name=short_name).config
    config.model_name = 'vit_large_patch16_224.mae'
    super().__init__(config=config)


@register('mae_large_image')
class MAELargeImage(MAEBaseVideo):

  def __init__(self, short_name: str | None = None):
    config = MAEBaseImage(short_name=short_name).config
    config.model_name = 'vit_large_patch16_224.mae'
    super().__init__(config=config)


@register('mae_huge_video')
class MAEHugeVideo(MAEBaseVideo):

  def __init__(self, short_name: str | None = None):
    config = MAEBaseVideo(short_name=short_name).config
    config.model_name = 'vit_huge_patch14_224.mae'
    super().__init__(config=config)


@register('mae_huge_image')
class MAEHugeImage(MAEBaseVideo):

  def __init__(self, short_name: str | None = None):
    config = MAEBaseImage(short_name=short_name).config
    config.model_name = 'vit_huge_patch14_224.mae'
    super().__init__(config=config)


@register('clip_base_video')
class CLIPBaseVideo(TimmImageAsVideoModelBase):

  def __init__(
      self, config: VideoModelConfig = None, short_name: str | None = None
  ):
    super().__init__(
        config
        if config is not None
        else VideoModelConfig(
            model_name='vit_base_patch16_clip_224.laion2b',
            short_name=short_name,
            clip_len=8,
            frame_sample_rate=4,
            native_num_frames=1,
        )
    )


@register('clip_base_image')
class CLIPBaseImage(CLIPBaseVideo):

  def __init__(self, short_name: str | None = None):
    config = CLIPBaseVideo(short_name=short_name).config
    config.clip_len = 1
    config.frame_sample_rate = 1
    super().__init__(config=config)


@register('clip_large_video')
class CLIPLargeVideo(CLIPBaseVideo):

  def __init__(self, short_name: str | None = None):
    config = CLIPBaseVideo(short_name=short_name).config
    config.model_name = 'vit_large_patch14_clip_224.laion2b'
    super().__init__(config=config)


@register('clip_large_image')
class CLIPLargeImage(CLIPBaseVideo):

  def __init__(self, short_name: str | None = None):
    config = CLIPBaseImage(short_name=short_name).config
    config.model_name = 'vit_large_patch14_clip_224.laion2b'
    super().__init__(config=config)


@register('clip_huge_video')
class CLIPHugeVideo(CLIPBaseVideo):

  def __init__(self, short_name: str | None = None):
    config = CLIPBaseVideo(short_name=short_name).config
    config.model_name = 'vit_huge_patch14_clip_224.laion2b'
    super().__init__(config=config)


@register('clip_huge_image')
class CLIPHugeImage(CLIPBaseVideo):

  def __init__(self, short_name: str | None = None):
    config = CLIPBaseImage(short_name=short_name).config
    config.model_name = 'vit_huge_patch14_clip_224.laion2b'
    super().__init__(config=config)


@register('clip_base_in12k_video')
class CLIPBaseIn12kVideo(CLIPBaseVideo):

  def __init__(self, short_name: str | None = None):
    config = CLIPBaseVideo(short_name=short_name).config
    config.model_name = 'vit_base_patch16_clip_224.laion2b_ft_in12k'
    super().__init__(config=config)


@register('clip_base_in12k_image')
class CLIPBaseIn12kImage(CLIPBaseVideo):

  def __init__(self, short_name: str | None = None):
    config = CLIPBaseImage(short_name=short_name).config
    config.model_name = 'vit_base_patch16_clip_224.laion2b_ft_in12k'
    super().__init__(config=config)


@register('clip_large_in12k_video')
class CLIPLargeIn12kVideo(CLIPBaseVideo):

  def __init__(self, short_name: str | None = None):
    config = CLIPBaseVideo(short_name=short_name).config
    config.model_name = 'vit_large_patch14_clip_224.laion2b_ft_in12k'
    super().__init__(config=config)


@register('clip_large_in12k_image')
class CLIPLargeIn12kImage(CLIPBaseVideo):

  def __init__(self, short_name: str | None = None):
    config = CLIPBaseImage(short_name=short_name).config
    config.model_name = 'vit_large_patch14_clip_224.laion2b_ft_in12k'
    super().__init__(config=config)


@register('clip_huge_in12k_video')
class CLIPHugeIn12kVideo(CLIPBaseVideo):

  def __init__(self, short_name: str | None = None):
    config = CLIPBaseVideo(short_name=short_name).config
    config.model_name = 'vit_huge_patch14_clip_224.laion2b_ft_in12k'
    super().__init__(config=config)


@register('clip_huge_in12k_image')
class CLIPHugeIn12kImage(CLIPBaseVideo):

  def __init__(self, short_name: str | None = None):
    config = CLIPBaseImage(short_name=short_name).config
    config.model_name = 'vit_huge_patch14_clip_224.laion2b_ft_in12k'
    super().__init__(config=config)


@register('pe_core_tiny_video')
class PerceptionEncoderCoreTiny(TimmImageAsVideoModelBase):

  def __init__(
      self, config: VideoModelConfig = None, short_name: str | None = None
  ):
    super().__init__(
        config
        if config is not None
        else VideoModelConfig(
            model_name='vit_pe_core_tiny_patch16_384',
            short_name=short_name,
            clip_len=8,
            frame_sample_rate=4,
            native_num_frames=1,
        )
    )

  def forward_intermediates(
      self, pixel_values: torch.Tensor, pool_type: str = 'cls'
  ) -> torch.Tensor:
    assert (
        pool_type in self.config.pool_types
    ), f'Only supported pool types: {self.config.pool_types}'
    self._ensure_initialized()
    pixel_values = pixel_values.to(self.device)
    b, f, c, h, w = pixel_values.shape
    pixel_values_reshaped = pixel_values.view(b * f, c, h, w)
    # TODO: test norm = True
    hidden_states = self.model.forward_intermediates(
        pixel_values_reshaped,
        output_fmt='NLC',
        intermediates_only=True,
        return_prefix_tokens=pool_type == 'cls',
        norm=True,
    )
    if pool_type == 'cls':
      # Second arg in tuple is the class token
      feats = [v[1].squeeze(1) for v in hidden_states]
    else:
      feats = [self.model.pool(v) for v in hidden_states]
    all_layer_features = torch.stack(feats, dim=1)
    all_layer_features = rearrange(
        all_layer_features, '(B F) L C -> B F L C', B=b
    )
    # By default average over all frames
    all_layer_features = all_layer_features.mean(dim=1)
    return all_layer_features


@register('pe_core_tiny_image')
class PerceptionEncoderCoreTinyImage(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTiny(short_name=short_name).config
    config.clip_len = 1
    config.frame_sample_rate = 1
    super().__init__(config=config)


@register('pe_core_small_video')
class PerceptionEncoderCoreSmall(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTiny(short_name=short_name).config
    config.model_name = 'vit_pe_core_small_patch16_384'
    super().__init__(config=config)


@register('pe_core_small_image')
class PerceptionEncoderCoreSmallImage(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTinyImage(short_name=short_name).config
    config.model_name = 'vit_pe_core_small_patch16_384'
    super().__init__(config=config)


@register('pe_core_base_video')
class PerceptionEncoderCoreBase(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTiny(short_name=short_name).config
    config.model_name = 'vit_pe_core_base_patch16_224'
    super().__init__(config=config)


@register('pe_core_base_image')
class PerceptionEncoderCoreBaseImage(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTinyImage(short_name=short_name).config
    config.model_name = 'vit_pe_core_base_patch16_224'
    super().__init__(config=config)


@register('pe_core_large_video')
class PerceptionEncoderCoreLarge(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTiny(short_name=short_name).config
    config.model_name = 'vit_pe_core_large_patch14_336'
    super().__init__(config=config)


@register('pe_core_large_image')
class PerceptionEncoderCoreLargeImage(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTinyImage(short_name=short_name).config
    config.model_name = 'vit_pe_core_large_patch14_336'
    super().__init__(config=config)


@register('pe_core_giant_video')
class PerceptionEncoderCoreGiant(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTiny(short_name=short_name).config
    config.model_name = 'vit_pe_core_gigantic_patch14_448'
    config.pool_types = ['avg']
    super().__init__(config=config)


@register('pe_core_giant_image')
class PerceptionEncoderCoreGiantImage(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTinyImage(short_name=short_name).config
    config.model_name = 'vit_pe_core_gigantic_patch14_448'
    config.pool_types = ['avg']
    super().__init__(config=config)


@register('pe_language_large_video')
class PerceptionEncoderLanguageLarge(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTiny(short_name=short_name).config
    config.model_name = 'vit_pe_lang_large_patch14_448'
    super().__init__(config=config)


@register('pe_language_large_image')
class PerceptionEncoderLanguageLargeImage(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTinyImage(short_name=short_name).config
    config.model_name = 'vit_pe_lang_large_patch14_448'
    super().__init__(config=config)


@register('pe_language_giant_video')
class PerceptionEncoderLanguageGiant(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTiny(short_name=short_name).config
    config.model_name = 'vit_pe_lang_gigantic_patch14_448'
    config.pool_types = ['avg']
    super().__init__(config=config)


@register('pe_language_giant_image')
class PerceptionEncoderLanguageGiantImage(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTinyImage(short_name=short_name).config
    config.model_name = 'vit_pe_lang_gigantic_patch14_448'
    config.pool_types = ['avg']
    super().__init__(config=config)


@register('pe_spatial_tiny_video')
class PerceptionEncoderSpatialTiny(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTiny(short_name=short_name).config
    config.model_name = 'vit_pe_spatial_tiny_patch16_512'
    super().__init__(config=config)


@register('pe_spatial_tiny_image')
class PerceptionEncoderSpatialTinyImage(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTinyImage(short_name=short_name).config
    config.model_name = 'vit_pe_spatial_tiny_patch16_512'
    super().__init__(config=config)


@register('pe_spatial_small_video')
class PerceptionEncoderSpatialSmall(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTiny(short_name=short_name).config
    config.model_name = 'vit_pe_spatial_small_patch16_512'
    super().__init__(config=config)


@register('pe_spatial_small_image')
class PerceptionEncoderSpatialSmallImage(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTinyImage(short_name=short_name).config
    config.model_name = 'vit_pe_spatial_small_patch16_512'
    super().__init__(config=config)


@register('pe_spatial_base_video')
class PerceptionEncoderSpatialBase(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTiny(short_name=short_name).config
    config.model_name = 'vit_pe_spatial_base_patch16_512'
    super().__init__(config=config)


@register('pe_spatial_base_image')
class PerceptionEncoderSpatialBaseImage(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTinyImage(short_name=short_name).config
    config.model_name = 'vit_pe_spatial_base_patch16_512'
    super().__init__(config=config)


@register('pe_spatial_large_video')
class PerceptionEncoderSpatialLarge(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTiny(short_name=short_name).config
    config.model_name = 'vit_pe_spatial_large_patch14_448'
    config.clip_len = 16
    config.frame_sample_rate = 2
    super().__init__(config=config)


@register('pe_spatial_large_image')
class PerceptionEncoderSpatialLargeImage(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTinyImage(short_name=short_name).config
    config.model_name = 'vit_pe_spatial_large_patch14_448'
    super().__init__(config=config)


@register('pe_spatial_giant_video')
class PerceptionEncoderSpatialGiant(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTiny(short_name=short_name).config
    config.model_name = 'vit_pe_spatial_gigantic_patch14_448'
    config.pool_types = ['avg']
    super().__init__(config=config)


@register('pe_spatial_giant_image')
class PerceptionEncoderSpatialGiantImage(PerceptionEncoderCoreTiny):

  def __init__(self, short_name: str | None = None):
    config = PerceptionEncoderCoreTinyImage(short_name=short_name).config
    config.model_name = 'vit_pe_spatial_gigantic_patch14_448'
    config.pool_types = ['avg']
    super().__init__(config=config)


@register('dinov3_base_video')
class DINOv3BaseVideo(HuggingFaceModelBase):

  def __init__(
      self, config: VideoModelConfig = None, short_name: str | None = None
  ):
    super().__init__(
        config
        if config is not None
        else VideoModelConfig(
            model_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
            short_name=short_name,
            clip_len=8,
            frame_sample_rate=4,
            native_num_frames=1,
        )
    )

  def forward_intermediates(
      self, pixel_values: torch.Tensor, pool_type: str = 'cls'
  ) -> torch.Tensor:
    """Returns [B, L, C] output, L = number of layers."""
    assert (
        pool_type in self.config.pool_types
    ), f'Only supported pool types: {self.config.pool_types}'
    self._ensure_initialized()
    pixel_values = pixel_values.to(self.device)
    b, f, c, h, w = pixel_values.shape
    pixel_values_reshaped = pixel_values.view(b * f, c, h, w)
    outputs = self.model(
        pixel_values=pixel_values_reshaped, output_hidden_states=True
    )
    # Drop the first embedding layer (not always true but shrug)
    hidden_states = outputs.hidden_states[1:]
    if pool_type == 'cls':
      feats = [v[:, 0, :] for v in hidden_states]
    elif pool_type == 'avg':
      feats = [v[:, 1:, :].mean(dim=1) for v in hidden_states]
    all_layer_features = torch.stack(feats, dim=1)
    all_layer_features = all_layer_features.view(
        b, f, -1, all_layer_features.shape[-1]
    ).mean(1)
    return all_layer_features


@register('dinov3_base_image')
class DINOv3BaseImage(DINOv3BaseVideo):

  def __init__(self, short_name: str | None = None):
    config = DINOv3BaseVideo(short_name=short_name).config
    config.clip_len = 1
    config.frame_sample_rate = 1
    super().__init__(config=config)


@register('dinov3_large_video')
class DINOv3LargeVideo(DINOv3BaseVideo):

  def __init__(self, short_name: str | None = None):
    config = DINOv3BaseVideo(short_name=short_name).config
    config.model_name = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
    super().__init__(config=config)


@register('dinov3_large_image')
class DINOv3LargeImage(DINOv3BaseVideo):

  def __init__(self, short_name: str | None = None):
    config = DINOv3BaseImage(short_name=short_name).config
    config.model_name = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
    super().__init__(config=config)


@register('dinov3_huge_video')
class DINOv3HugeVideo(DINOv3BaseVideo):

  def __init__(self, short_name: str | None = None):
    config = DINOv3BaseVideo(short_name=short_name).config
    config.model_name = 'facebook/dinov3-vith16plus-pretrain-lvd1689m'
    super().__init__(config=config)


@register('dinov3_huge_image')
class DINOv3HugeImage(DINOv3BaseVideo):

  def __init__(self, short_name: str | None = None):
    config = DINOv3BaseImage(short_name=short_name).config
    config.model_name = 'facebook/dinov3-vith16plus-pretrain-lvd1689m'
    super().__init__(config=config)


@register('dinov3_7b_video')
class DINOv37BVideo(DINOv3BaseVideo):

  def __init__(self, short_name: str | None = None):
    config = DINOv3BaseVideo(short_name=short_name).config
    config.model_name = 'facebook/dinov3-vit7b16-pretrain-lvd1689m'
    super().__init__(config=config)


@register('dinov3_7b_image')
class DINOv37BImage(DINOv3BaseVideo):

  def __init__(self, short_name: str | None = None):
    config = DINOv3BaseImage(short_name=short_name).config
    config.model_name = 'facebook/dinov3-vit7b16-pretrain-lvd1689m'
    super().__init__(config=config)


@register('webssl_dino1b_224_video')
class WebSSLDINO1b224(DINOv3BaseVideo):

  def __init__(
      self, config: VideoModelConfig = None, short_name: str | None = None
  ):
    super().__init__(
        config
        if config is not None
        else VideoModelConfig(
            model_name='facebook/webssl-dino1b-full2b-224',
            short_name=short_name,
            clip_len=8,
            frame_sample_rate=4,
            native_num_frames=1,
        )
    )

  def _load_model(self):
    return Dinov2Model.from_pretrained(
        self.config.model_name, attn_implementation='sdpa'
    )


@register('webssl_dino1b_224_image')
class WebSSLDINO1b224Image(WebSSLDINO1b224):

  def __init__(self, short_name: str | None = None):
    config = WebSSLDINO1b224(short_name=short_name).config
    config.clip_len = 1
    config.frame_sample_rate = 1
    super().__init__(config=config)


@register('webssl_dino5b_224_video')
class WebSSLDINO5b224(WebSSLDINO1b224):

  def __init__(self, short_name: str | None = None):
    config = WebSSLDINO1b224(short_name=short_name).config
    config.model_name = 'facebook/webssl-dino5b-full2b-224'
    super().__init__(config=config)


@register('webssl_dino5b_224_image')
class WebSSLDINO5b224Image(WebSSLDINO1b224):

  def __init__(self, short_name: str | None = None):
    config = WebSSLDINO1b224Image(short_name=short_name).config
    config.model_name = 'facebook/webssl-dino5b-full2b-224'
    super().__init__(config=config)


@register('webssl_dino7b_224_video')
class WebSSLDINO7b224(WebSSLDINO1b224):

  def __init__(self, short_name: str | None = None):
    config = WebSSLDINO1b224(short_name=short_name).config
    config.model_name = 'facebook/webssl-dino7b-full8b-224'
    super().__init__(config=config)


@register('webssl_dino7b_224_image')
class WebSSLDINO7b224Image(WebSSLDINO1b224):

  def __init__(self, short_name: str | None = None):
    config = WebSSLDINO1b224Image(short_name=short_name).config
    config.model_name = 'facebook/webssl-dino7b-full8b-224'
    super().__init__(config=config)


@register('vjepa2_large')
class VJEPA2Large256(HuggingFaceModelBase):

  def __init__(
      self, config: VideoModelConfig = None, short_name: str | None = None
  ):
    super().__init__(
        config
        if config is not None
        else VideoModelConfig(
            model_name='facebook/vjepa2-vitl-fpc64-256',
            short_name=short_name,
            clip_len=16,
            frame_sample_rate=2,
            native_num_frames=1,
            pool_types=['avg'],
        )
    )

  def _load_raw_processor(self):
    return AutoVideoProcessor.from_pretrained(self.config.model_name)

  def preprocess(self, images: list[np.ndarray]) -> torch.Tensor:
    self._ensure_initialized()
    inputs = self._raw_processor(videos=images, return_tensors='pt')
    return inputs['pixel_values_videos']

  def forward_intermediates(
      self, pixel_values: torch.Tensor, pool_type: str = 'avg'
  ) -> torch.Tensor:
    """Returns [B, L, C] output, L = number of layers."""
    assert (
        pool_type in self.config.pool_types
    ), f'Only supported pool types: {self.config.pool_types}'
    self._ensure_initialized()
    pixel_values = pixel_values.to(self.device)
    outputs = self.model(
        pixel_values_videos=pixel_values, output_hidden_states=True
    )
    hidden_states = outputs.hidden_states[1:]
    # only avg
    feats = [v[:, 1:, :].mean(dim=1) for v in hidden_states]
    all_layer_features = torch.stack(feats, dim=1)
    return all_layer_features


@register('vjepa2_huge')
class VJEPA2Huge256(VJEPA2Large256):

  def __init__(self, short_name: str | None = None):
    config = VJEPA2Large256(short_name=short_name).config
    config.model_name = 'facebook/vjepa2-vith-fpc64-256'
    super().__init__(config=config)


@register('vjepa2_giant')
class VJEPA2Giant256(VJEPA2Large256):

  def __init__(self, short_name: str | None = None):
    config = VJEPA2Large256(short_name=short_name).config
    config.model_name = 'facebook/vjepa2-vitg-fpc64-256'
    super().__init__(config=config)


@register('vjepa2_giant_384')
class VJEPA2Giant384(VJEPA2Large256):

  def __init__(self, short_name: str | None = None):
    config = VJEPA2Large256(short_name=short_name).config
    config.model_name = 'facebook/vjepa2-vitg-fpc64-384'
    super().__init__(config=config)


if __name__ == '__main__':
  import pprint

  pprint.pp(list(VIDEO_MODEL_REGISTRY.keys()))
  print(f'{len(VIDEO_MODEL_REGISTRY)} available models.')

  for model_name, my_model in VIDEO_MODEL_REGISTRY.items():
    print(f'\n--- Testing {model_name} ---')

    my_model.to('cuda')  # Initialize and move to device
    dummy_frames = [
        np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
        for _ in range(my_model.config.clip_len)
    ]

    # Test call
    processed_frames = my_model.preprocess(dummy_frames)
    print(processed_frames.shape)
    outputs = my_model(processed_frames.unsqueeze(0), pool_type='avg')
    print(outputs.shape)
    my_model.cleanup()
