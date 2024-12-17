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

"""Structured Log of Pipeline Parallelism Benchmark Results.

Typical usage example:

  log = PPBenchmarkResultsLog(metadata)
  for batch_idx in range(metadata.n_batch):
    for microbatch_idx in range(metadata.n_microbatch):
      # Instrument timing experiment 
      timestamps_ns = do_timing_experiment()
      log.record_microbatch_comm(batch_idx, microbatch_idx, timestamps_ns)
    log.record_barrier_time()
  log.save_results(output_dir)
"""

import pathlib
import time
# 
from typing import List, Protocol

from google.protobuf import text_format
import torch

import straggler_detection_healthcheck_pb2


class TimeSource(Protocol):
  """Defines an interface for a PPBenchmark timing source."""

  def time_ns(self) -> int:
    ...

  def perf_counter_ns(self) -> int:
    ...


class PPBenchmarkResultsLog:
  """Log of PPBenchmark Results.

  This log is responsible for recording & exporting the results
  of PPBenchmark tests.
  """

  def __init__(
      self,
      metadata: straggler_detection_healthcheck_pb2.Metadata,
      time_source: TimeSource = time,
  ):
    """Constructor for PPBenchmarkResultsLog.

    Args:
      metadata: Metadata relating to the current run.
      time_source: Source of timing data. Can be overridden for testing.
    """
    self._time_source = time_source
    self._metadata = metadata
    self._barrier_perf_counter_ns = None  # perf_counter_ns at last run barrier
    self._barrier_time_ns = None  # time_ns at last run barrier
    # Pre-allocate data array as a performance optimization.
    self._data = [straggler_detection_healthcheck_pb2.PPBenchmarkResult()] * (
        self._metadata.n_batch * self._metadata.n_microbatch
    )
    self._data_counter = 0

  def record_microbatch_comm(
      self, batch_idx: int, microbatch_idx: int, timestamps_ns: List[int]
  ) -> None:
    """Record data for each microbatch communication.

    Args:
      batch_idx: index of the batch
      microbatch_idx: index of the microbatch
      timestamps_ns: list of four timestamps, in nanoseconds, corresponding to
        the following events: - t0: before starting async transfers - t1: after
        starting async transfer but before work.wait() - t2: after work.wait()
        but before cuda.synchronize() - t3: after cuda.synchronize()
    """
    barrier_relative_timestamps_ns = [
        ts - self._barrier_perf_counter_ns for ts in timestamps_ns
    ]
    self._data[self._data_counter] = (
        straggler_detection_healthcheck_pb2.PPBenchmarkResult(
            batch_id=batch_idx,
            microbatch_id=microbatch_idx,
            barrier_time_ns=self._barrier_time_ns,  # epoch ns of last barrier
            t0_ns=barrier_relative_timestamps_ns[0],  # t0 (barrier offset ns)
            t1_ns=barrier_relative_timestamps_ns[1],  # t1 (barrier offset ns)
            t2_ns=barrier_relative_timestamps_ns[2],  # t2 (barrier offset ns)
            t3_ns=barrier_relative_timestamps_ns[3],  # t3 (barrier offset ns)
        )
    )
    self._data_counter += 1

  def record_barrier_time(self):
    """Syncronizes all processes and records the current time."""
    torch.cuda.synchronize()
    torch.distributed.barrier()
    self._barrier_time_ns = self._time_source.time_ns()
    self._barrier_perf_counter_ns = self._time_source.perf_counter_ns()

  def get_results(
      self,
  ) -> straggler_detection_healthcheck_pb2.PPBenchmarkResults:
    """Packages tracked timing logs into a PPBenchmarkResults proto.

    Returns:
      PPBenchmarkResults proto containing the results of the benchmark.
    """
    return straggler_detection_healthcheck_pb2.PPBenchmarkResults(
        metadata=self._metadata,
        benchmark_results=self._data,
    )

  def save_results(self, output_dir: str) -> None:
    """Saves results to a textproto in the output directory.

    Args:
      output_dir: directory to save the results to
    """

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    file_path = f"{output_dir}/{int(self._metadata.rank)}_rank-{int(self._metadata.msg_size_mb)}_mb.textproto"
    with open(file_path, "w") as f:
      f.write(text_format.MessageToString(self.get_results()))
