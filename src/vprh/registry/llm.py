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

"""LLM model registry from huggingface."""

from dataclasses import dataclass
import pprint

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import T5EncoderModel
from vprh.misc.utils import auto_determine_dtype


@dataclass
class LLMConfig:
  """Generic config for LLM settings."""

  model_name: str
  short_name: str | None = None
  pad_token: str | None = None


LLM_REGISTRY = {
    'bloom-560m': LLMConfig(
        model_name='bigscience/bloomz-560m', short_name='bloom-560m'
    ),
    'bloom-1b1': LLMConfig(
        model_name='bigscience/bloomz-1b1', short_name='bloom-1b1'
    ),
    'bloom-1b7': LLMConfig(
        model_name='bigscience/bloomz-1b7', short_name='bloom-1b7'
    ),
    'bloom-3b': LLMConfig(
        model_name='bigscience/bloomz-3b', short_name='bloom-3b'
    ),
    'bloom-7b1': LLMConfig(
        model_name='bigscience/bloomz-7b1', short_name='bloom-7b1'
    ),
    'open-llama-3b': LLMConfig(
        model_name='openlm-research/open_llama_3b', short_name='open-llama-3b'
    ),
    'open-llama-7b': LLMConfig(
        model_name='openlm-research/open_llama_7b', short_name='open-llama-7b'
    ),
    'open-llama-13b': LLMConfig(
        model_name='openlm-research/open_llama_13b', short_name='open-llama-13b'
    ),
    'llama-7b': LLMConfig(
        model_name='huggyllama/llama-7b',
        short_name='llama-7b',
        pad_token='[PAD]',
    ),
    'llama-13b': LLMConfig(
        model_name='huggyllama/llama-13b',
        short_name='llama-13b',
        pad_token='[PAD]',
    ),
    'llama-30b': LLMConfig(
        model_name='huggyllama/llama-30b',
        short_name='llama-30b',
        pad_token='[PAD]',
    ),
    'llama-65b': LLMConfig(
        model_name='huggyllama/llama-65b',
        short_name='llama-65b',
        pad_token='[PAD]',
    ),
    'gemma-2b': LLMConfig(model_name='google/gemma-2b', short_name='gemma-2b'),
    'gemma-7b': LLMConfig(model_name='google/gemma-7b', short_name='gemma-7b'),
    'gemma2-2b': LLMConfig(
        model_name='google/gemma-2-2b', short_name='gemma2-2b'
    ),
    'gemma2-2b-it': LLMConfig(
        model_name='google/gemma-2-2b-it', short_name='gemma2-2b-it'
    ),
    'gemma2-9b': LLMConfig(
        model_name='google/gemma-2-9b', short_name='gemma2-9b'
    ),
    'gemma2-9b-it': LLMConfig(
        model_name='google/gemma-2-9b-it', short_name='gemma2-9b-it'
    ),
    'gemma2-27b': LLMConfig(
        model_name='google/gemma-2-27b', short_name='gemma2-27b'
    ),
    'gemma2-27b-it': LLMConfig(
        model_name='google/gemma-2-27b-it', short_name='gemma2-27b-it'
    ),
    'gemma3-1b-pt': LLMConfig(
        model_name='google/gemma-3-1b-pt', short_name='gemma3-1b-pt'
    ),
    'gemma3-1b-it': LLMConfig(
        model_name='google/gemma-3-1b-it', short_name='gemma3-1b-it'
    ),
    'gemma3-4b-pt': LLMConfig(
        model_name='google/gemma-3-4b-pt', short_name='gemma3-4b-pt'
    ),
    'gemma3-4b-it': LLMConfig(
        model_name='google/gemma-3-4b-it', short_name='gemma3-4b-it'
    ),
    'gemma3-12b-pt': LLMConfig(
        model_name='google/gemma-3-12b-pt', short_name='gemma3-12b-pt'
    ),
    'gemma3-12b-it': LLMConfig(
        model_name='google/gemma-3-12b-it', short_name='gemma3-12b-it'
    ),
    'gemma3-27b-pt': LLMConfig(
        model_name='google/gemma-3-27b-pt', short_name='gemma3-27b-pt'
    ),
    'gemma3-27b-it': LLMConfig(
        model_name='google/gemma-3-27b-it', short_name='gemma3-27b-it'
    ),
    't5-small': LLMConfig(
        model_name='google-t5/t5-small', short_name='t5-small'
    ),
    't5-base': LLMConfig(model_name='google-t5/t5-base', short_name='t5-base'),
    't5-large': LLMConfig(
        model_name='google-t5/t5-large', short_name='t5-large'
    ),
    't5-3b': LLMConfig(model_name='google-t5/t5-3b', short_name='t5-3b'),
    't5-11b': LLMConfig(model_name='google-t5/t5-11b', short_name='t5-11b'),
    'llama3.2-1b': LLMConfig(
        model_name='meta-llama/Llama-3.2-1B-Instruct', short_name='llama3.2-1b'
    ),
    'llama3.2-3b': LLMConfig(
        model_name='meta-llama/Llama-3.2-3B-Instruct', short_name='llama3.2-3b'
    ),
    'llama3.1-8b': LLMConfig(
        model_name='meta-llama/Llama-3.1-8B-Instruct', short_name='llama3.1-8b'
    ),
    'llama3.3-70b': LLMConfig(
        model_name='meta-llama/Llama-3.3-70B-Instruct',
        short_name='llama3.3-70b',
    ),
}


def create_llm(cfg: LLMConfig, quantize=False, force_download=False):
  """Instantiates an LLM from a given config."""
  compute_dtype, torch_dtype = auto_determine_dtype()

  quantization_config = None
  if quantize:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )

  if 't5' in cfg.model_name:
    language_model = T5EncoderModel.from_pretrained(
        cfg.model_name, output_hidden_states=True, device_map='cuda'
    )
  else:
    language_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map='auto',
        quantization_config=quantization_config,
        dtype=torch_dtype,
        force_download=force_download,
        output_hidden_states=True,
    )

  language_model = language_model.eval()
  return language_model


def create_tokenizer(cfg: LLMConfig):
  """Setting up tokenizer."""
  tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

  if cfg.pad_token:
    tokenizer.pad_token = cfg.pad_token
  elif tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  tokenizer.padding_side = 'left'
  return tokenizer


if __name__ == '__main__':
  pprint.pp(list(LLM_REGISTRY.keys()))
