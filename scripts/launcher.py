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

"""A launcher script for parallel feature extraction across multiple GPUs."""

import dataclasses
from datetime import datetime
import os
import signal
import subprocess
import sys
import time

from registry import CONFIGS
import torch
import tyro
from vprh.configs.args import ScriptArgs as WorkerScriptArgs


@tyro.conf.configure(tyro.conf.ConsolidateSubcommandArgs)
@dataclasses.dataclass
class LauncherArgs(WorkerScriptArgs):
  """Arguments for launching the worker script."""

  num_gpus: int = 1
  """The number of GPUs to parallelize the inference over."""


# --- SIGNAL HANDLING SETUP ---
# Make this a global list so our handler function can access it
worker_processes = []


def signal_handler(*args, **kwargs):
  """Handles termination signals with a two-phase shutdown: 1.

  Asks all workers to terminate gracefully (SIGTERM). 2. If they don't exit
  within a timeout, it forces them to exit (SIGKILL).
  """
  shutdown_timeout = 10  # Seconds to wait before forcing shutdown

  print(
      '\nCaught termination signal. Asking workers to shut down gracefully...'
  )

  # --- Phase 1: Terminate Politely ---
  for proc in worker_processes:
    proc.terminate()

  start_time = time.time()

  # --- Phase 2: Wait and Forcefully Kill if Necessary ---
  while time.time() - start_time < shutdown_timeout:
    # Check if all processes have terminated
    if all(proc.poll() is not None for proc in worker_processes):
      print('All workers shut down gracefully.')
      sys.exit(0)
    time.sleep(0.5)  # Wait a bit before checking again

  # If we reach here, the timeout was exceeded
  print(f'Timeout of {shutdown_timeout}s exceeded. Forcing worker shutdown...')
  for proc in worker_processes:
    # poll() returns None if the process is still running
    if proc.poll() is None:
      print(f'  - Killing stubborn worker (PID: {proc.pid}).')
      proc.kill()

  print('All workers have been terminated. Exiting.')
  sys.exit(0)


# --- END SIGNAL HANDLING SETUP ---


if __name__ == '__main__':
  # Register our custom handler to be called on SIGINT (Ctrl+C)
  signal.signal(signal.SIGINT, signal_handler)

  args = tyro.extras.overridable_config_cli(CONFIGS)

  # Logging setup
  if len(sys.argv) > 1 and sys.argv[1] in CONFIGS:
    config_name = sys.argv[1]
  else:
    config_name = 'default'

  timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  log_dir = os.path.join('logs', f'{config_name}_{timestamp}')

  try:
    os.makedirs(log_dir, exist_ok=True)
    print(f'Logs for this run will be saved in: {log_dir}')
  except OSError as e:
    print(f'Error creating log directory {log_dir}: {e}')
    sys.exit(1)

  log_files = []

  worker_command = ['uv', 'run', 'scripts/main_extract.py', *sys.argv[1:]]
  if '--num-gpus' not in worker_command:
    worker_command.extend(['--num-gpus', str(args.num_gpus)])

  print(
      f'Launcher starting {args.num_gpus} worker processes...(Press Ctrl+C to'
      ' terminate all)'
  )

  available_devices_str = os.environ.get('CUDA_VISIBLE_DEVICES')
  if available_devices_str:
    available_devices = available_devices_str.split(',')
  else:
    available_devices = [str(i) for i in range(torch.cuda.device_count())]

  if len(available_devices) < args.num_gpus:
    raise ValueError(
        f'Not enough available devices! Requesting {args.num_gpus}, but only '
        f'{len(available_devices)} are available: {available_devices}'
    )

  for i in range(args.num_gpus):
    env = os.environ.copy()
    gpu_id = available_devices[i]
    env['CUDA_VISIBLE_DEVICES'] = gpu_id
    cmd = [*worker_command, f'--gpu-id={i}']

    log_file_path = os.path.join(log_dir, f'gpu_{gpu_id}.log')
    print(f'  - Worker {i}: Logs will be written to {log_file_path}')

    log_file = open(log_file_path, 'w')
    log_files.append(log_file)

    proc = subprocess.Popen(
        cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT
    )
    # Add the newly created process to our global list
    worker_processes.append(proc)

  # Wait for all worker processes to complete (normal execution path)
  for proc in worker_processes:
    proc.wait()

  # Cleanup
  for f in log_files:
    f.close()

  print(
      'All worker processes have completed successfully. Logs are saved in'
      f' {log_dir}.'
  )
