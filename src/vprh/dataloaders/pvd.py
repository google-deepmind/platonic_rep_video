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

"""Dataloader for the Perception Video Dataset (PVD)."""

import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder
from vprh.dataloaders.vatex import sample_frame_indices
from vprh.dataloaders.vatex import temporal_resize


def create_pvd_captions(file_path):
  """Creates a dataset list from annotation jsonl.

  Args:
      file_path (str): The path to annotation jsonl.

  Returns:
      list: A list of dictionaries containing all captions.
  """
  dataset = []

  # Just use the entirety of the caption for now
  with open(file_path) as f:
    annotations = [json.loads(line) for line in f.readlines()]

  # Include all of the different options for captions, pick later
  for ex in annotations:
    data = {
        'description': ex['description'],
        'human_caption': ex['human_caption'],
        'keywords': ex['keywords'],
        'model_caption': ex['model_caption'],
    }
    if 'rephrased_captions' in ex:
      data['rephrased_captions'] = ex['rephrased_captions']
    dataset.append(data)
  return dataset


class PerceptionVideoDataset(Dataset):
  """Dataset for Perception Video Dataset (PVD)."""

  def __init__(
      self,
      annotation_path: str,
      data_dir: str,
      image_transform=None,
      process_mode='center',
  ):
    """Initializes a PVD dataset.

    Args:
        annotation_path (str): jsonl file to processed pvd dataset
        data_dir (str): base path for videos
        image_transform (_type_, optional): Image transform if including videos.
          Defaults to None.
        process_mode (str, optional): The video sampling mode. Defaults to
          'center'.
    """
    self.data_dir = Path(data_dir)
    with open(annotation_path) as f:
      self.annotations = [json.loads(line) for line in f.readlines()]

    self.image_transform = image_transform
    self.clip_len = None
    self.frame_sample_rate = None
    self.process_mode = process_mode
    self.native_num_frames = None

  def __len__(self):
    """Returns the number of data points in the dataset."""
    return len(self.annotations)

  def __getitem__(self, idx) -> dict[str, Any]:
    """Gets PVD dataset item.

    Returns:
        Dict[str, Any]: with keys
            pixel_values (torch.Tensor): values of the video frames
    """
    example = self.annotations[idx]

    payload = {}
    if self.image_transform is not None:
      video_path = self.data_dir / f'{example["video_id"]}.bin'
      video = VideoDecoder(video_path, dimension_order='NHWC')
      if 'clip_start_frame' in example:
        start, end = example['clip_start_frame'], example['clip_end_frame']
      else:
        start, end = 0, min(int(example['num_frames']), len(video) - 1)

      indices = sample_frame_indices(
          self.clip_len,
          start,
          end,
          self.frame_sample_rate,
          mode=self.process_mode,
      )
      raw_frames = video.get_frames_at(indices).data.numpy()

      if (
          self.native_num_frames is not None
          and self.clip_len % self.native_num_frames != 0
      ):
        factor = self.clip_len // self.native_num_frames + 1
        raw_frames = temporal_resize(
            raw_frames, factor * self.native_num_frames
        )

      frames = [
          self.image_transform(Image.fromarray(frame)).squeeze()
          for frame in raw_frames
      ]
      payload['pixel_values'] = torch.stack(frames)
      del video

    return payload

  def update_transforms(self, image_transform=None):
    """Updates the image transform for the dataset.

    Args:
        image_transform: The new image transform to use.
    """
    self.image_transform = image_transform

  def update_video_args(
      self, clip_len=None, frame_sample_rate=None, native_num_frames=None
  ):
    """Updates video processing arguments.

    Args:
        clip_len: The number of frames to sample.
        frame_sample_rate: The rate at which to sample frames.
        native_num_frames: The native number of frames for the model.
    """
    self.clip_len = clip_len
    self.frame_sample_rate = frame_sample_rate
    self.native_num_frames = native_num_frames

  def clear(self):
    """Clears the dataset's video-specific arguments and transforms."""
    self.image_transform = None
    self.clip_len = None
    self.frame_sample_rate = None
