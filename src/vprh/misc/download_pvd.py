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

try:
  from datasets import Video  # pylint: disable=g-import-not-at-top
except ImportError:  # Older `datasets` releases have no Video feature.
  Video = None


def _mp4_bytes(mp4) -> bytes:
  """Returns the raw mp4 bytes of a dataset row.

  Depending on the `datasets` version the column is raw bytes, or a dict with
  `bytes` and/or `path` when decoding is disabled.

  Args:
    mp4: The value of the `mp4` column.
  """
  if isinstance(mp4, (bytes, bytearray)):
    return bytes(mp4)
  if mp4.get('bytes'):
    return mp4['bytes']
  with open(mp4['path'], 'rb') as f:
    return f.read()


def _write_atomic(path: Path, data: bytes):
  """Writes `data` to `path` so an interrupted download leaves no partial file."""
  tmp = path.with_name(path.name + '.part')
  tmp.write_bytes(data)
  tmp.rename(path)


def main():
  """Main function to download videos."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--local_json_path', '-l', type=Path, default=None)
  parser.add_argument('--output_dir', '-o', type=Path, default='data/pvd')
  args = parser.parse_args()

  ds = load_dataset('facebook/PE-Video', split='test', streaming=True)
  if Video is not None:
    # Keep the encoded mp4 bytes: recent `datasets` versions would otherwise
    # decode them with torchcodec, which may not be loadable on every machine.
    ds = ds.cast_column('mp4', Video(decode=False))
  video_path = Path(args.output_dir)
  video_path.mkdir(exist_ok=True, parents=True)

  if args.local_json_path is not None:
    print(f'Downloading videos from {args.local_json_path}')
    with open(args.local_json_path) as f:
      video_ids = {str(json.loads(line)['video_id']) for line in f}

    # Resume: skip videos that were already downloaded.
    already_downloaded = {
        p.stem for p in video_path.glob('*.bin') if p.stat().st_size > 0
    }
    video_ids_to_find = video_ids - already_downloaded
    print(
        f'{len(video_ids)} videos wanted, '
        f'{len(video_ids & already_downloaded)} already on disk'
    )
    with tqdm(total=len(video_ids_to_find), desc='Downloading videos') as pbar:
      for row in ds:
        if not video_ids_to_find:
          break
        key = row['__key__']
        if key in video_ids_to_find:
          _write_atomic(video_path / f'{key}.bin', _mp4_bytes(row['mp4']))
          pbar.update(1)
          video_ids_to_find.remove(key)
    if video_ids_to_find:
      raise SystemExit(
          f'{len(video_ids_to_find)} videos were not found in the dataset, '
          f'e.g. {sorted(video_ids_to_find)[:3]}'
      )
  else:
    print('Downloading all videos from PE Video dataset')
    with open('pe_video_dataset.json', 'w') as f:
      f.write('')

    for row in tqdm(ds):
      key = row['__key__']
      _write_atomic(video_path / f'{key}.bin', _mp4_bytes(row['mp4']))

      with open('pe_video_dataset.json', 'a') as f:
        f.write(json.dumps(row['json']) + '\n')


if __name__ == '__main__':
  main()
