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

"""Dynamically registers experiment configurations."""

import importlib.util
import os
import sys


def load_configs_from_path(configs_path: str, src_path: str) -> dict:
  """Dynamically loads configurations from a specified directory.

  Args:
    configs_path: The absolute path to the directory containing config files.
    src_path: The absolute path to the source directory to be added to the
      python path, allowing configs to import project modules.

  Returns:
    A dictionary mapping configuration names to their loaded config objects.
  """
  loaded_configs = {}

  # Add the src directory to Python's path to allow imports from 'vprh'
  if src_path not in sys.path:
    sys.path.insert(0, src_path)

  for filename in os.listdir(configs_path):
    if filename.endswith('.py') and not filename.startswith('__'):
      module_name = filename[:-3]
      file_path = os.path.join(configs_path, filename)

      try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not (spec and spec.loader):
          continue

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, 'config'):
          loaded_configs[module_name] = module.config
      except Exception as e:
        print(f"Warning: Could not load config '{module_name}': {e}")

  return loaded_configs


# --- Main execution ---
# Define paths relative to this script's location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC_PATH = os.path.join(_PROJECT_ROOT, 'src')
_CONFIGS_PATH = os.path.join(_PROJECT_ROOT, 'configs')

# Load all configurations into the global CONFIGS dictionary
CONFIGS = load_configs_from_path(_CONFIGS_PATH, _SRC_PATH)
