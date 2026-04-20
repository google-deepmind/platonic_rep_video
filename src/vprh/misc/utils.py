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

"""Miscellaneous utility functions."""

import math
import os

import torch


def to_feature_filename(
    output_dir,
    dataset,
    model_name,
    pool,
    num_captions=-1,
    sample_mode='linspace',
    caption_key='',
    num_frames=-1,
):
  """Constructs a filename for saving feature files.

  Args:
      output_dir: The directory to save the file in.
      dataset: The name of the dataset.
      model_name: The name of the model.
      pool: The pooling strategy used.
      num_captions: The number of captions used.
      sample_mode: The video frame sampling mode.
      caption_key: The key for captions in the dataset.
      num_frames: The number of frames sampled.
  """
  save_name = f'{model_name.replace("/", "_")}'

  save_name += f'_pool-{pool}'
  if sample_mode != 'linspace':
    save_name += f'_sample-{sample_mode}'
  if num_captions > -1:
    save_name += f'_numcap-{num_captions}'
  if caption_key != '' and 'captions' not in caption_key:
    save_name += f'_capkey-{caption_key}'
  if num_frames > -1:
    save_name += f'_numframes-{num_frames}'

  save_path = os.path.join(output_dir, dataset, f'{save_name}.pt')
  return save_path


def to_alignment_filename(
    output_dir,
    dataset,
    modality_x,
    pool_x,
    prompt_x,
    modality_y,
    pool_y,
    prompt_y,
    metric,
    topk,
    num_captions=-1,
):
  """Constructs a filename for saving alignment score files.

  Args:
      output_dir: The directory to save the file in.
      dataset: The name of the dataset.
      modality_x: The name of the first modality.
      pool_x: The pooling strategy for the first modality.
      prompt_x: The prompt used for the first modality.
      modality_y: The name of the second modality.
      pool_y: The pooling strategy for the second modality.
      prompt_y: The prompt used for the second modality.
      metric: The alignment metric used.
      topk: The top-k value for k-NN metrics.
      num_captions: The number of captions used.
  """
  caption_sffx = '' if num_captions == -1 else f'_cid-{num_captions}'
  save_path = os.path.join(
      output_dir,
      dataset,
      f'{modality_x}_pool-{pool_x}_prompt-{prompt_x}_{modality_y}_pool-{pool_y}_prompt-{prompt_y}'
      + caption_sffx,
      (
          f'{metric}_k{topk}.npy'
          if 'knn' in metric or metric == 'mutual_witness_complex'
          else f'{metric}.npy'
      ),
  )
  return save_path


def cross_entropy_loss(llm_inputs, llm_outputs):
  """Computes the cross-entropy loss for a batch of LLM outputs.

  Args:
      llm_inputs: A dictionary containing the input IDs and attention mask.
      llm_outputs: The output from the language model, containing logits.
  """
  criterion = torch.nn.CrossEntropyLoss(reduction='none')
  mask = llm_inputs['attention_mask'][:, :-1]
  loss = mask * criterion(
      llm_outputs['logits'][:, :-1].permute(0, 2, 1),
      llm_inputs['input_ids'][:, 1:],
  )
  avg_loss = loss.sum(-1) / mask.sum(-1)
  return loss, avg_loss


def cross_entropy_to_bits_per_unit(losses, input_strings, unit='byte'):
  """Convert cross-entropy losses from nats to bits per unit.

  Parameters:
  - losses (torch.Tensor): [batch x seq_len] (padding tokens should be 0)
  - input_strings (list of str): List of original input strings.
  - unit (str): The unit for normalization ('byte' or 'char').

  Returns:
  - torch.Tensor: Tensor of bits per byte values, one per input string.
  """  # noqa: E501
  # nats to bits by multiplying with log base 2 of e
  # sum over the sequence length (total bits for each input string)
  losses_in_bits = (losses.cpu() * torch.log2(torch.tensor(math.e))).sum(1)
  # Calculate bytes for each input string and normalize losses
  # (8 bits per character, so roughly num character * 8)
  if unit == 'byte':
    bytes_per_input = torch.tensor(
        [len(s.encode('utf-8')) for s in input_strings], dtype=torch.float32
    )
  elif unit == 'char':
    bytes_per_input = torch.tensor(
        [len(s) for s in input_strings], dtype=torch.float32
    )
  else:
    raise ValueError(f'Unsupported unit {unit}')

  # mormalize by the total number of bytes per input string
  bits_per_byte = losses_in_bits / bytes_per_input
  return bits_per_byte


def auto_determine_dtype():
  """Automatic dtype setting.

  override this if you want to force a specific dtype
  """
  compute_dtype = torch.bfloat16 if check_bfloat16_support() else torch.float32
  torch_dtype = torch.bfloat16 if check_bfloat16_support() else torch.float32
  print(f'compute_dtype:\t{compute_dtype}')
  print(f'torch_dtype:\t{torch_dtype}')
  return compute_dtype, torch_dtype


def check_bfloat16_support():
  """Checks if cuda driver/device supports bfloat16 computation."""
  if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    compute_capability = torch.cuda.get_device_capability(current_device)
    if compute_capability[0] >= 8:  # Check if device supports bfloat16
      return True
    else:
      return False
  else:
    return None
