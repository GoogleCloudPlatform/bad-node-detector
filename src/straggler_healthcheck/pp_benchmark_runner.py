# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pipeline Parallelism Benchmark Runner.

Bridges the gap between user inputs & the pipeline parallelism benchmark,
responsible for:
  - Gathering all user/machine inputs
  - Enforcing input formatting & constraints
  - Initializing the process group
  - Running the benchmark
"""

import os
import socket

from absl import app
from absl import flags
import torch.distributed as dist

import pp_benchmark

_MESSAGE_SIZES_MB = flags.DEFINE_list(
    "message_sizes_mb",
    ["10"],
    "The message sizes (in MB) to use for the benchmark. Default: [10]",
)

_OMPI_COMM_WORLD_RANK = flags.DEFINE_integer(
    "ompi_comm_world_rank",
    None,
    "The rank of the process.",
    required=True,
)

_OMPI_COMM_WORLD_LOCAL_RANK = flags.DEFINE_string(
    "ompi_comm_world_local_rank",
    None,
    "The local rank of the process.",
    required=True,
)

_MAIN_ADDRESS = flags.DEFINE_string(
    "main_address",
    None,
    "The main address of the process.",
    required=True,
)

_HOSTNAME = flags.DEFINE_string(
    "hostname",
    None,
    "The hostname of the process.",
    required=True,
)

_N_NODES = flags.DEFINE_integer(
    "n_nodes",
    None,
    "The number of nodes to use for the benchmark.",
    required=True,
)

_N_GPUS_PER_NODE = flags.DEFINE_integer(
    "n_gpus_per_node",
    8,
    "The number of GPUs per node to use for the benchmark. Default: 8",
)

_N_BATCH = flags.DEFINE_integer(
    "n_batch",
    50,
    "The number of batches to use for the benchmark. Default: 50",
)

_N_MICROBATCH = flags.DEFINE_integer(
    "n_microbatch",
    50,
    "The number of microbatches to use for the benchmark. Default: 50",
)

_BIDIRECTIONAL = flags.DEFINE_bool(
    "bidirectional",
    False,
    "Whether to use bidirectional communication for the benchmark. Default:"
    " False",
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "The output directory to use for the benchmark. Default: None",
    required=True,
)


def main(_) -> None:
  message_sizes_mb = [
      int(msg_size_mb_str) for msg_size_mb_str in (_MESSAGE_SIZES_MB.value)
  ]
  device_id = _OMPI_COMM_WORLD_LOCAL_RANK.value
  os.environ["CUDA_VISIBLE_DEVICES"] = device_id

  init_method_hostname = _MAIN_ADDRESS.value
  init_method = f"tcp://{init_method_hostname}:2379"

  n_nodes = _N_NODES.value
  n_gpus_per_node = _N_GPUS_PER_NODE.value
  world_size = n_nodes * n_gpus_per_node
  world_rank = _OMPI_COMM_WORLD_RANK.value

  print("initializing process group")
  dist.init_process_group(
      backend="nccl",
      rank=world_rank,
      world_size=world_size,
      init_method=init_method,
  )

  rank = dist.get_rank()

  machine = socket.getaddrinfo(socket.gethostname(), None)
  if _HOSTNAME.value:
    hostname = _HOSTNAME.value
  else:
    hostname = socket.gethostname()
  print(
      f"rank: {rank}, local_rank: {device_id} server: {machine}, hostname:"
      f" {hostname}"
  )

  for message_size_mb in message_sizes_mb:
    pp_benchmark.run_pp_benchmark(
        hostname=hostname,
        output_dir=_OUTPUT_DIR.value,
        message_size_mb=message_size_mb,
        n_gpus_per_node=n_gpus_per_node,
        n_nodes=n_nodes,
        n_batch=_N_BATCH.value,
        n_microbatch=_N_MICROBATCH.value,
        bidirectional=_BIDIRECTIONAL.value,
    )


if __name__ == "__main__":
  app.run(main)
