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

"""Extracts features from video and language models.

Adapted from the original PRH code.
https://github.com/minyoungg/platonic-rep/tree/main
"""

import gc
import os

import torch
from torch.utils import data
from tqdm import tqdm
from tqdm import trange
from vprh.configs import args as args_module
from vprh.misc import utils
from vprh.registry import llm as llm_registry
from vprh.registry import video as video_registry


def extract_llm_features(
    llm_cfg: llm_registry.LLMConfig,
    dataset: list[dict],
    args: args_module.ScriptArgs,
):
  """Extracts features from language models.

  Args:
      llm_cfg: language model config
      dataset: huggingface dataset or dictionary
      args: argparse arguments
  """
  caption_key = args.llm_caption_key
  num_captions = args.num_captions

  if caption_key in dataset[0]:
    if num_captions > -1:
      texts = [' '.join(ex[caption_key][:num_captions]) for ex in dataset]
    else:
      texts = [' '.join(ex[caption_key]) for ex in dataset]
  else:
    raise ValueError(
        f'Caption key {caption_key} not found in dataset ({dataset[0].keys()}).'
    )

  print(f'Found {len(texts)} texts in the dataset.')
  print(f'Example: {texts[0]}')

  save_path = utils.to_feature_filename(
      args.feature_dir,
      args.dataset,
      llm_cfg.short_name,
      pool=args.llm_pool,
      num_captions=num_captions,
      caption_key=caption_key,
  )

  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  print(f'Dataset:\t{args.dataset}')
  print(f'Processing:\t{llm_cfg.short_name}')
  print(f'Save path:\t{save_path}')

  if os.path.exists(save_path) and not args.force_remake:
    print('File exists. Skipping.\n', flush=True)
    return

  print(f'Loading tokenizer for {llm_cfg.short_name}', flush=True)
  tokenizer = llm_registry.create_tokenizer(llm_cfg)
  print('Done.')

  print(f'Loading language model for {llm_cfg.short_name}', flush=True)
  language_model = llm_registry.create_llm(
      llm_cfg, quantize=args.quantize, force_download=args.force_download
  )

  llm_param_count = sum([p.numel() for p in language_model.parameters()])
  print(f'LLM param count: {llm_param_count / 1e6:.2f}M')

  tokens = tokenizer(texts, padding='longest', return_tensors='pt')
  llm_feats, losses, bpb_losses = [], [], []

  device = next(language_model.parameters()).device
  print(f'Using device: {device}')
  tokens = {k: v.to(device) for k, v in tokens.items()}

  for i in trange(0, len(texts), args.batch_size):
    token_inputs = {
        k: v[i : i + args.batch_size].to(device).long()
        for (k, v) in tokens.items()
    }

    with torch.no_grad():
      if 'olmo' in llm_cfg.model_name.lower():
        llm_output = language_model(
            input_ids=token_inputs['input_ids'],
            attention_mask=token_inputs['attention_mask'],
            output_hidden_states=True,
        )
      else:
        llm_output = language_model(
            input_ids=token_inputs['input_ids'],
            attention_mask=token_inputs['attention_mask'],
        )

      if 't5' not in llm_cfg.model_name:
        loss, avg_loss = utils.cross_entropy_loss(token_inputs, llm_output)
        bpb = utils.cross_entropy_to_bits_per_unit(
            loss.cpu(), texts[i : i + args.batch_size], unit='byte'
        )
        losses.extend(avg_loss.cpu())
        bpb_losses.extend(bpb)
      else:
        losses.append(torch.tensor(0.0))
        bpb_losses.append(torch.tensor(0.0))

      if args.llm_pool == 'avg':
        feats = torch.stack(llm_output['hidden_states']).permute(1, 0, 2, 3)
        mask = token_inputs['attention_mask'].unsqueeze(-1).unsqueeze(1)
        feats = (feats * mask).sum(2) / mask.sum(2)
      elif args.llm_pool == 'last':
        feats = [v[:, -1, :] for v in llm_output['hidden_states']]
        feats = torch.stack(feats).permute(1, 0, 2)
      else:
        raise NotImplementedError(f'Unknown pooling {args.llm_pool}')
      llm_feats.append(feats.cpu())

  if losses:
    print(f'Average loss: {torch.stack(losses).mean().item()}\n')

  save_dict = {
      'feats': torch.cat(llm_feats).cpu(),
      'num_params': llm_param_count,
      'mask': tokens['attention_mask'].cpu(),
  }
  if losses:
    save_dict['loss'] = torch.stack(losses).mean()
  if bpb_losses:
    save_dict['bpb'] = torch.stack(bpb_losses).mean()

  torch.save(save_dict, save_path)

  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
  gc.collect()


def extract_video_features(
    video_model: video_registry.VideoModelInterface,
    dataset: data.Dataset,
    args: args_module.ScriptArgs,
):
  """Extracts features from vision models.

  Args:
      video_model: The video model interface.
      dataset: The dataset to extract features from.
      args: argparse arguments
  """
  short_name = video_model.config.short_name
  video_cfg = video_model.config

  if args.video_pool not in video_cfg.pool_types:
    print(
        f'Overriding pooling for {video_cfg.short_name} to'
        f' {video_cfg.pool_types[0]}'
    )
    video_pool = video_cfg.pool_types[0]
  else:
    video_pool = args.video_pool

  # ------ Saving ----------
  save_path = utils.to_feature_filename(
      args.feature_dir,
      args.dataset,
      short_name,
      pool=video_pool,
      sample_mode=args.video_sample_mode,
      num_frames=args.num_frames,
  )
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  print(f'Dataset: \t{args.dataset}')
  print(f'Video_model_name:\t{short_name}')
  print(f'Save_path: \t{save_path}')

  if os.path.exists(save_path) and not args.force_remake:
    print('file exists. skipping')
    return
  elif args.force_remake:
    print('File ', save_path, ' EXISTS but remaking')
  else:
    print('File ', save_path, ' DOES NOT exist')

  # ------ Instantiate model --------
  video_model.to('cuda' if torch.cuda.is_available() else 'cpu')

  # ------ Create Dataloader -------
  dataset.update_transforms(image_transform=video_model.preprocess)
  # If no provided num_frames, set to clip_len
  if args.num_frames == -1:
    dataset.update_video_args(
        clip_len=video_cfg.clip_len,
        native_num_frames=video_cfg.native_num_frames,
        frame_sample_rate=video_cfg.frame_sample_rate,
    )
  else:
    dataset.update_video_args(
        clip_len=args.num_frames,
        native_num_frames=video_cfg.clip_len,
        frame_sample_rate=video_cfg.frame_sample_rate,
    )
  dataloader = data.DataLoader(
      dataset,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=min(10, os.cpu_count() or 1),
      prefetch_factor=2,
  )

  video_model_param_count = sum(
      [p.numel() for p in video_model.model.parameters()]
  )
  print(f'Params: {video_model_param_count / 1e6} M')

  video_feats = []
  good = []
  print('dataset length = ', len(dataset))

  base_num_frames = video_cfg.clip_len
  for batch in tqdm(dataloader):
    processed_frames = batch['pixel_values']
    b, f, h, w, c = processed_frames.shape
    if f > base_num_frames:
      processed_frames = processed_frames.reshape(-1, base_num_frames, h, w, c)

    with torch.no_grad():
      outputs = video_model(processed_frames, video_pool)

    # If more frames were processed than the model's native clip_len,
    # it implies multiple segments were created from the input.
    # We average features across these segments.
    if f > base_num_frames:
      outputs = outputs.reshape(b, -1, *outputs.shape[1:]).mean(dim=1)

    video_feats.append(outputs)

  video_feats = torch.cat(video_feats)
  torch.save(
      {
          'feats': video_feats,
          'num_params': video_model_param_count,
          'good': good,
      },
      save_path,
  )
  video_model.cleanup()
  # Clear video model specific settings from above
  dataset.clear()
  return
