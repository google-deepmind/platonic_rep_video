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

"""Configuration for a sample VaTeX experiment."""

from vprh.configs.args import ScriptArgs


config = (
    'Sample VaTeX experiment.',
    ScriptArgs(
        feature_dir='./results/sample',
        alignment_dir='./results/alignment',
        dataset='vatex',
        annotation_path='vatex.data.img_png.pkl',
        batch_size=4,
        video_pool='cls',
        video_sample_mode='linspace',
        llm_names=[
            'gemma2-2b-it',
            'gemma2-9b-it',
        ],
        video_model_names=[
            'videomaev2_base',
            'videomaev2_large',
            'dinov2_large_video',
            'dinov2_large_image',
        ],
        num_captions=-1,
        num_frames=-1,
        force_download=False,
    ),
)
