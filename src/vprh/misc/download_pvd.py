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

"""Downloads videos from the Perception Video Dataset (PVD)."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
  """Main function to download videos."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--local_json_path', '-l', type=Path, default=None)
  parser.add_argument('--output_dir', '-o', type=Path, default='data/pvd')
  args = parser.parse_args()

  ds = load_dataset('facebook/PE-Video', split='test', streaming=True)
  video_path = Path(args.output_dir)
  video_path.mkdir(exist_ok=True, parents=True)

  if args.local_json_path is not None:
    print(f'Downloading videos from {args.local_json_path}')
    with open(args.local_json_path) as f:
      video_ids = {str(json.loads(line)['video_id']) for line in f}

    video_ids_to_find = video_ids.copy()
    with tqdm(total=len(video_ids), desc='Downloading videos') as pbar:
      for row in ds:
        if not video_ids_to_find:
          break
        key = row['__key__']
        if key in video_ids_to_find:
          with open(video_path / f'{key}.bin', 'wb') as f:
            f.write(row['mp4'])
          pbar.update(1)
          video_ids_to_find.remove(key)
  else:
    print('Downloading all videos from PE Video dataset')
    with open('pe_video_dataset.json', 'w') as f:
      f.write('')

    for row in tqdm(ds):
      key = row['__key__']
      with open(video_path / f'{key}.bin', 'wb') as f:
        f.write(row['mp4'])

      with open('pe_video_dataset.json', 'a') as f:
        f.write(json.dumps(row['json']) + '\n')


if __name__ == '__main__':
  main()
