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

"""Dataloader for the VaTeX dataset."""

from io import BytesIO
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def temporal_resize(video: np.ndarray, num_output_frames: int) -> np.ndarray:
  """Temporally resizes a video using nearest-neighbor interpolation."""
  assert video.ndim == 4, 'Video must have 4 input dimensions!'
  t, h, w, c = video.shape

  if t == num_output_frames:
    return video

  # Generate indices of frames to pick
  indices = np.linspace(0, t - 1, num_output_frames)
  indices = np.round(indices).astype(int)

  resized_video = video[indices]

  assert resized_video.shape == (num_output_frames, h, w, c)
  return resized_video


def sample_frame_indices(
    num_frames, start_frame, end_frame, frame_sample_rate, mode='linspace'
):
  """Samples a given number of frame indices from a video range.

  Args:
      num_frames (int): Total number of frames to sample.
      start_frame (int): The starting index of the available video range
        (inclusive).
      end_frame (int): The ending index of the available video range
        (exclusive).
      frame_sample_rate (int): The interval between frames (temporal stride).
        Used only for 'random' and 'center' modes to determine clip length.
      mode (str, optional): Sampling strategy. Defaults to "linspace". Available
        options: - 'random': Randomly selects a contiguous sequence of frames
          based on frame_sample_rate. - 'center': Selects a sequence centered in
          the video range based on frame_sample_rate. - 'linspace': Samples
          frames evenly spaced across the entire range (ignores
          frame_sample_rate). - 'linspace_center': Divides the range into
          segments and samples the center of each segment.

  Returns:
      np.ndarray: A 1D array of size [num_frames] containing the integer
      indices.

  Note:
      If the required duration (num_frames * frame_sample_rate) exceeds the
      available frames (end_frame - start_frame), the function falls back to
      'linspace' mode automatically.

  Raises:
      ValueError: If num_frames is larger than the total available frames.
      ValueError: If an unknown mode is provided.
  """
  if num_frames > end_frame - start_frame:
    raise ValueError(
        f'Video is not long enough for {num_frames} '
        f'(only {end_frame - start_frame})'
    )

  clip_frame_len = int(num_frames * frame_sample_rate)
  if clip_frame_len > end_frame - start_frame:
    print('Falling back to linspace sampling, not enough frames!')
    print(
        f'Requested {clip_frame_len} frames, only'
        f' {end_frame - start_frame} available.'
    )
    mode = 'linspace'

  if mode == 'random':
    start_idx = np.random.randint(start_frame, end_frame - clip_frame_len - 1)
    indices = np.linspace(
        start_idx, start_idx + clip_frame_len, num_frames, dtype=int
    )
  elif mode == 'center':
    center = (start_frame + end_frame) // 2
    start_idx = center - clip_frame_len // 2
    indices = np.linspace(
        start_idx, start_idx + clip_frame_len, num_frames, dtype=int
    )
  elif mode == 'linspace':
    # OOD for video models!
    indices = np.linspace(start_frame, end_frame - 1, num=num_frames, dtype=int)
  elif mode == 'linspace_center':
    indices, step = np.linspace(
        start_frame,
        end_frame - 1,
        num=num_frames,
        dtype=int,
        endpoint=False,
        retstep=True,
    )
    indices += round(step) // 2
  else:
    raise ValueError(f'Unknown sampling mode {mode}')

  return indices


def create_vatex_captions(file_path):
  """Creates a dataset list from annotation jsonl.

  Args:
      file_path (str): The path to annotation jsonl.

  Returns:
      list: A list of dictionaries containing all captions.
  """
  dataset = []

  with open(file_path) as f:
    annotations = [json.loads(line) for line in f.readlines()]

  for ex in annotations:
    dataset.append({'captions': ex['captions']})

  return dataset


class VaTeX(Dataset):
  """Dataset for VaTeX."""

  def __init__(
      self, video_path: str, image_transform=None, process_mode='random'
  ):
    """Initializes a VaTeX dataset.

    Args:
        video_path (str): path to a numpy array of processed videos
        image_transform (_type_, optional): Image transform if including videos.
          Defaults to None.
        process_mode (str, optional): The video sampling mode. Defaults to
          'random'.
    """
    self.video_path = Path(video_path)
    with open(video_path, 'rb') as f:
      self.data_points = pickle.load(f)  # Load data from the pickle file

    print('VaTeX dataset created')

    self.image_transform = image_transform
    self.clip_len = None
    self.frame_sample_rate = None
    self.process_mode = process_mode

  def __len__(self):
    """Returns the number of data points in the dataset."""
    return len(self.data_points)

  def __getitem__(self, idx) -> dict[str, Any]:
    """Gets VaTeX dataset item.

    Returns:
        Dict[str, Any]: with keys
            pixel_values (torch.Tensor): values of the video frames

    Note:
        Our videos are stored as bytes in a pickle file, as we preprocess
        our data to 48 sampled frames.
        This likely isn't useful for you; Refer to the PVD loader for a
        more traditional setup, or modify for your own uses.
    """
    example = self.data_points[idx]

    payload = {}
    start = 0
    end = len(example['image'])

    indices = sample_frame_indices(
        self.clip_len,
        start,
        end,
        self.frame_sample_rate,
        mode=self.process_mode,
    )

    frames = []
    for frame_idx in indices:
      assert frame_idx < len(example['image']), 'Index out of bounds.'
      image_data = example['image'][frame_idx]
      image = Image.open(BytesIO(image_data)).convert('RGB')
      frames.append(np.array(image))  # Convert to numpy array

    if (
        self.native_num_frames is not None
        and self.clip_len % self.native_num_frames != 0
    ):
      factor = self.clip_len // self.native_num_frames + 1
      frames = temporal_resize(
          np.stack(frames), factor * self.native_num_frames
      )

    if self.image_transform:
      frames = [
          self.image_transform(Image.fromarray(frame)).squeeze()
          for frame in frames
      ]
      payload['pixel_values'] = torch.stack(frames)
    else:
      payload['pixel_values'] = torch.Tensor(frames)

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
    self.native_num_frames = None
