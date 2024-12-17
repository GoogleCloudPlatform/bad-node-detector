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

"""Implementation of the Pipeline Parallelism Benchmark.

This benchmark is used to instrument communication timing information between
GPUs across Nodes. Communication originates from the current Node (depth) and
GPU pairing.

e.g. Take a 3 node pipeline with 4 GPUs per node. If global rank is 4, then we'd
currently be running on the first GPU of the Second Node (see below).

  Node 0:  0   1  2  3
  Node 1: [0]  1  2  3
  Node 2:  0   1  2  3

This benchmark will run communication between the other GPU 0s - sending to
Node 2, and receiving from Node 0:

  Node 0: 0  1  2  3
          ↓
  Node 1: 0  1  2  3
          ↓
  Node 2: 0  1  2  3

The time it takes to complete this communication is recorded in batches and
saved for later analysis.
"""

import contextlib
import functools
import gc
import time
# 
from typing import List, MutableSequence, Optional

import torch
import torch.distributed as dist

import pp_benchmark_results_log
import straggler_detection_healthcheck_pb2

_KILOBYTE = 1024


def _schedule_transfers(
    depth_idx: int,
    tensor_idx: int,
    dst_rank: int,
    src_rank: int,
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
) -> MutableSequence[dist.Work]:
  """Schedules async transfers between source and destination GPUs.

  Args:
    depth_idx: The depth index of the current GPU.
    tensor_idx: The index of the tensor to send/receive.
    dst_rank: The rank of the destination GPU to send to.
    src_rank: The rank of the source GPU to receive from.
    send_tensor: The tensor to send.
    recv_tensor: The tensor to receive.

  Returns:
    A list of async transfer handles.
  """
  work_handles = []
  if depth_idx % 2 == 0:
    if dst_rank is not None:
      work_handles.append(
          dist.isend(tensor=send_tensor[tensor_idx], dst=dst_rank)
      )
    if src_rank is not None:
      work_handles.append(
          dist.irecv(tensor=recv_tensor[tensor_idx], src=src_rank)
      )
  else:
    if src_rank is not None:
      work_handles.append(
          dist.irecv(tensor=recv_tensor[tensor_idx], src=src_rank)
      )
    if dst_rank is not None:
      work_handles.append(
          dist.isend(tensor=send_tensor[tensor_idx], dst=dst_rank)
      )
  return work_handles


def _get_tensor_id(microbatch_idx: int, tensor_size: int) -> int:
  """Returns the index of the tensor to send/receive in the current microbatch.

  Args:
    microbatch_idx: The index of the current microbatch.
    tensor_size: The number of tensors to send/receive.

  Returns:
    The index of the tensor to send/receive in the current microbatch.
  """
  return microbatch_idx % (tensor_size - 1)


def _do_microbatch_comm(
    depth_idx: int,
    microbatch_idx: int,
    send_tensor_fwd: torch.Tensor,
    recv_tensor_fwd: torch.Tensor,
    send_tensor_bwd: torch.Tensor,
    recv_tensor_bwd: torch.Tensor,
    next_rank: int,
    prev_rank: int,
    bidirectional: bool,
) -> List[int]:
  """Performs a single set of communication between GPUs.

  Args:
    depth_idx: The depth index of the current GPU.
    microbatch_idx: The microbatch index of the current communication.
    send_tensor_fwd: The tensor to send forward.
    recv_tensor_fwd: The tensor to receive forward.
    send_tensor_bwd: The tensor to send backward.
    recv_tensor_bwd: The tensor to receive backward.
    next_rank: The rank of the next GPU in the pipeline.
    prev_rank: The rank of the previous GPU in the pipeline.
    bidirectional: Whether to run bidirectional communication.

  Returns:
    A tuple of timing events corresponding to:
      - Immediately before starting async transfers
      - Immediately after starting async transfers
      - Immediately after completing async transfers
      - Immediately after synchronizing the current GPU
  """
  tensor_idx = _get_tensor_id(microbatch_idx, len(send_tensor_fwd))
  timestamps = []

  snapshot_time = lambda: timestamps.append(time.perf_counter_ns())

  snapshot_time()
  work_handles = _schedule_transfers(
      depth_idx=depth_idx,
      tensor_idx=tensor_idx,
      dst_rank=next_rank,
      src_rank=prev_rank,
      send_tensor=send_tensor_fwd,
      recv_tensor=recv_tensor_fwd,
  )
  if bidirectional:
    work_handles.extend(
        _schedule_transfers(
            depth_idx=depth_idx,
            tensor_idx=tensor_idx,
            dst_rank=prev_rank,
            src_rank=next_rank,
            send_tensor=send_tensor_bwd,
            recv_tensor=recv_tensor_bwd,
        )
    )
  snapshot_time()

  _ = [handle.wait() for handle in work_handles]
  snapshot_time()
  torch.cuda.synchronize()
  snapshot_time()
  return timestamps


def _get_depth(global_rank: int, n_gpus_per_node: int) -> int:
  """Returns the depth index for the current node.

  Args:
    global_rank: The global rank of the current GPU.
    n_gpus_per_node: The number of GPUs per node.
  """
  return global_rank // n_gpus_per_node


def _get_gpu_idx(rank: int, depth: int, n_gpus_per_node: int) -> int:
  """Returns the GPU index for the current node.

  Args:
    rank: The rank of the current node.
    depth: The depth index of the current GPU.
    n_gpus_per_node: The number of GPUs per node.
  """
  return rank - depth * n_gpus_per_node


def _get_next_rank(
    depth: int, gpu_idx: int, n_nodes: int, n_gpus_per_node: int
) -> Optional[int]:
  """Returns the rank of the next GPU in the pipeline.

  Args:
    depth: The depth index of the current GPU.
    gpu_idx: The GPU index of the current GPU.
    n_nodes: The number of nodes in the pipeline.
    n_gpus_per_node: The number of GPUs per node.

  Returns:
    The rank of the next GPU in the pipeline, or None if given the last node.
  """
  return (
      None if depth == n_nodes - 1 else (depth + 1) * n_gpus_per_node + gpu_idx
  )


def _get_prev_rank(
    depth: int, gpu_idx: int, n_gpus_per_node: int
) -> Optional[int]:
  """Returns the rank of the last GPU in the pipeline.

  Args:
    depth: The depth index of the current GPU.
    gpu_idx: The GPU index of the current GPU.
    n_gpus_per_node: The number of GPUs per node.

  Returns:
    The rank of the last GPU in the pipeline, or None if given the first node.
  """
  return None if depth == 0 else (depth - 1) * n_gpus_per_node + gpu_idx


def _get_num_elements(message_size_mb: int, element_size_bytes: int) -> int:
  """Returns the number of elements in the message to send/receive.

  Args:
    message_size_mb: The size of the message to send/receive in megabytes.
    element_size_bytes: The size of each element in the message in bytes.
  """
  return message_size_mb * _KILOBYTE * _KILOBYTE // element_size_bytes


def run_pp_benchmark(
    hostname: str,
    message_size_mb: int,
    n_gpus_per_node: int,
    n_nodes: int,
    output_dir: str,
    n_batch: int = 10,
    n_microbatch: int = 500,
    n_warmup_runs: int = 10,
    bidirectional: bool = False,
) -> None:
  """Runs PPBenchmark on the current node and outputs results to a file.

  Args:
    hostname: The hostname of the current node.
    message_size_mb: The size of the messages to send and receive between GPUs.
    n_gpus_per_node: The number of GPUs per node.
    n_nodes: The number of nodes in the pipeline.
    output_dir: The directory to output the results to.
    n_batch: The number of batches to run.
    n_microbatch: The number of microbatches per batch. Must be >= 20.
    n_warmup_runs: The number of warmup runs to run.
    bidirectional: Whether to run bidirectional communication.
  """
  my_rank = dist.get_rank()
  my_depth_idx = _get_depth(my_rank, n_gpus_per_node)
  my_gpu_idx = _get_gpu_idx(my_rank, my_depth_idx, n_gpus_per_node)
  next_rank = _get_next_rank(my_depth_idx, my_gpu_idx, n_nodes, n_gpus_per_node)
  prev_rank = _get_prev_rank(my_depth_idx, my_gpu_idx, n_gpus_per_node)

  results_log = pp_benchmark_results_log.PPBenchmarkResultsLog(
      straggler_detection_healthcheck_pb2.Metadata(
          hostname=hostname,
          rank=my_rank,
          prev_rank=prev_rank,
          next_rank=next_rank,
          node_id=my_depth_idx,
          gpu_id=my_gpu_idx,
          n_batch=n_batch,
          n_microbatch=n_microbatch,
          msg_size_mb=message_size_mb,
      ),
  )

  element_size_bytes = 2  # 2 bytes per element (torch.bfloat16)
  n_elements = _get_num_elements(message_size_mb, element_size_bytes)
  randn = functools.partial(torch.rand, dtype=torch.bfloat16, device="cuda")
  gc.disable()

  build_comm_tensor = lambda n_elements, n_microbatch: [
      randn(n_elements) for _ in range(min(n_microbatch, 20) + 1)
  ]
  send_tensor_fwd = build_comm_tensor(n_elements, n_microbatch)
  recv_tensor_fwd = build_comm_tensor(n_elements, n_microbatch)
  send_tensor_bwd = (
      build_comm_tensor(n_elements, n_microbatch) if bidirectional else None
  )
  recv_tensor_bwd = (
      build_comm_tensor(n_elements, n_microbatch) if bidirectional else None
  )

  # Warm Up
  results_log.record_barrier_time()
  for _ in range(n_warmup_runs):
    _do_microbatch_comm(
        depth_idx=my_depth_idx,
        microbatch_idx=n_microbatch,
        send_tensor_fwd=send_tensor_fwd,
        recv_tensor_fwd=recv_tensor_fwd,
        send_tensor_bwd=send_tensor_bwd,
        recv_tensor_bwd=recv_tensor_bwd,
        next_rank=next_rank,
        prev_rank=prev_rank,
        bidirectional=bidirectional,
    )
  results_log.record_barrier_time()

  for batch_idx in range(n_batch):
    context = contextlib.nullcontext()
    with context:
      torch.cuda.synchronize()
      for microbatch_idx in range(n_microbatch):
        ts = _do_microbatch_comm(
            depth_idx=my_depth_idx,
            microbatch_idx=microbatch_idx,
            send_tensor_fwd=send_tensor_fwd,
            recv_tensor_fwd=recv_tensor_fwd,
            send_tensor_bwd=send_tensor_bwd,
            recv_tensor_bwd=recv_tensor_bwd,
            next_rank=next_rank,
            prev_rank=prev_rank,
            bidirectional=bidirectional,
        )
        results_log.record_microbatch_comm(batch_idx, microbatch_idx, ts)
      results_log.record_barrier_time()
  results_log.save_results(output_dir=output_dir)
