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

"""Post-process and analyze Pipeline Parallelism benchmark data.

Provides functions to read, preprocess, and plot Pipeline Parallelism benchmark
data.

Example usage:
  experiment_name = "experiment_name"
  # A directory containing files of the form ".*{experiment_name}.textproto"
  data_dir = "/path/to/benchmark/data"
  output_dir = "/tmp/out"
  # include events immediately before and after a known straggler
  interesting_event_offset = 1
  straggler_interesting_latency_ms = 5

  experiment_data = read_experiment_data(data_dir, experiment_name)
  heatmap_data = preprocess_experiment_data(
      experiment_data,
      straggler_threshold_ms,
      interesting_event_offset,
  )
  if heatmap_data is not None:
    output_file = os.path.join(output_dir, f"{experiment_name}.svg")
    plot_straggler_heatmap(heatmap_data, output_file)
"""

import dataclasses
import os
import pathlib
# 
from typing import Callable, List, Optional, Set

from google.protobuf import text_format
from matplotlib import colors
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import straggler_detection_healthcheck_pb2

_NANOSECONDS_PER_MILLISECOND = 1000000


def _read_pp_benchmark_data(
    data_dir, data_file
) -> straggler_detection_healthcheck_pb2.PPBenchmarkResults:
  """Read PP Benchmark Results from a textproto file.

  Args:
    data_dir: directory containing Pipeline Parallelism benchmark results
    data_file: name of the textproto file

  Returns:
    A PPBenchmarkResults proto containing the benchmark results.
  """
  pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
  with open(os.path.join(data_dir, data_file)) as f:
    results = text_format.Parse(
        f.read(), straggler_detection_healthcheck_pb2.PPBenchmarkResults()
    )
  return results


def _extract_send_recv_durations_for_experiment(
    benchmark_results: straggler_detection_healthcheck_pb2.PPBenchmarkResults,
    duration_extractor: Callable[
        [straggler_detection_healthcheck_pb2.PPBenchmarkResult], int
    ] = lambda result: (result.t3_ns - result.t0_ns),
) -> npt.NDArray[int]:
  """Extract SendRecv durations for all benchmark results as Milliseconds.

  Args:
    benchmark_results: Results for a given Pipeline Parallelism experiment
    duration_extractor: Function to extract the duration of a SendRecv event.
      Defaults to (t3 - t0)

  Returns:
    An array of SendRecv durations for all benchmark results in Milliseconds.
  """
  return np.array(
      [
          duration_extractor(result) // _NANOSECONDS_PER_MILLISECOND
          for result in benchmark_results.benchmark_results
      ],
      dtype="int",
  )


def _identify_interesting_event_indices(
    send_recv_timing_results: npt.NDArray[int],
    straggler_threshold_ms: int,
    interesting_event_offset: int,
) -> Set[int]:
  """Identify interesting event indices for a given experiment.

  Args:
    send_recv_timing_results: SendRecv durations for all benchmark results in
      Milliseconds
    straggler_threshold_ms: threshold for marking an event as a straggler
    interesting_event_offset: The fwd and bwd offset from a known straggler for
      marking an event as interesting

  Returns:
    A set of event indices that are deemed 'interesting'.
  """
  interesting_events = set()
  max_num_events = len(send_recv_timing_results)
  for event_idx, sr_time_ms in enumerate(send_recv_timing_results):
    if sr_time_ms >= straggler_threshold_ms:
      for offset in range(interesting_event_offset + 1):
        if event_idx - offset >= 0:
          interesting_events.add(event_idx - offset)
        if event_idx + offset < max_num_events:
          interesting_events.add(event_idx + offset)

  return interesting_events


@dataclasses.dataclass
class HeatmapData:
  """Data for plotting a heatmap."""

  experiment_name: str
  x_vals: List[int]
  y_labels: List[str]
  max_rank: int
  max_node_idx: int
  straggler_threshold_ms: int
  # A matrix of shape (max_rank, len(x_vals)) where each row
  # represents a GPU-Node pair and each column represents an event. The value
  # at each index represents the latency (ms) t3 - t2 for the corresponding
  # event.
  delayed_event_matrix: npt.NDArray[int]


def read_experiment_data(
    data_dir: str, experiment: str
) -> List[straggler_detection_healthcheck_pb2.PPBenchmarkResults]:
  """Read all benchmark results for a given experiment.

  Args:
    data_dir: directory containing Pipeline Parallelism benchmark results
    experiment: name of the experiment

  Returns:
    A list of PPBenchmarkResults for the given experiment found in the data_dir.
  """
  benchmark_results = []
  for filename in os.listdir(data_dir):
    if filename.endswith(f"{experiment}.textproto"):  # for each rank
      benchmark_results.append(_read_pp_benchmark_data(data_dir, filename))
  return benchmark_results


def preprocess_experiment_data(
    benchmark_results: List[
        straggler_detection_healthcheck_pb2.PPBenchmarkResults
    ],
    experiment: str,
    straggler_threshold_ms: int,
    interesting_event_offset: int,
) -> Optional[HeatmapData]:
  """Produces data needed to plot a heatmap for a given experiment.

  Args:
    benchmark_results: Results for a given Pipeline Parallelism experiment
    experiment: name of the experiment
    straggler_threshold_ms: threshold for marking an event as a straggler
    interesting_event_offset: Fwd and bwd offset from a known straggler for
      marking an event as interesting

  Returns:
    A HeatmapData object containing the data needed to plot a heatmap. Returns
    None if there are no interesting events.
  """

  send_recv_duration_ms_per_device: dict[tuple[int, int], npt.NDArray[int]] = (
      dict()
  )
  interesting_events = set()
  max_rank = 0
  max_node_idx = 0
  for pp_benchmark in benchmark_results:
    max_rank = max(max_rank, pp_benchmark.metadata.rank)
    max_node_idx = max(max_node_idx, pp_benchmark.metadata.node_id)
    # 
    benchmark_id = (pp_benchmark.metadata.gpu_id, pp_benchmark.metadata.node_id)
    send_recv_duration_ms_per_device[benchmark_id] = (
        _extract_send_recv_durations_for_experiment(pp_benchmark)
    )

    interesting_events_for_experiment = _identify_interesting_event_indices(
        send_recv_timing_results=send_recv_duration_ms_per_device[benchmark_id],
        straggler_threshold_ms=straggler_threshold_ms,
        interesting_event_offset=interesting_event_offset,
    )
    interesting_events.update(interesting_events_for_experiment)

  if not interesting_events:
    print(f"No interesting events found for {experiment}")
    return None

  x_vals = list(sorted(interesting_events))

  ordered_keys_for_heatmap = sorted(send_recv_duration_ms_per_device.keys())
  # Make a matrix focusing on slow events. Exclude fast events.
  delay_matrix = np.array([
      [send_recv_duration_ms_per_device[key][event] for event in x_vals]
      for key in ordered_keys_for_heatmap
  ])
  y_labels = [
      f"gpu-{gpu_idx}-node-{node_idx}"
      for (gpu_idx, node_idx) in ordered_keys_for_heatmap
  ]
  return HeatmapData(
      experiment_name=experiment,
      max_node_idx=max_node_idx,
      max_rank=max_rank,
      x_vals=x_vals,
      y_labels=y_labels,
      delayed_event_matrix=delay_matrix,
      straggler_threshold_ms=straggler_threshold_ms,
  )


@dataclasses.dataclass
class HeatmapMetadata:
  """Metadata for plotting a Straggler Heatmap."""

  min_width: int
  max_width: int
  min_height: int
  max_height: int


def plot_straggler_heatmap(
    heatmap_data: HeatmapData,
    output_file: str,
    heatmap_metadata: HeatmapMetadata = HeatmapMetadata(
        min_width=16, max_width=50, min_height=6, max_height=24
    ),
) -> plt.Figure:
  """Plots a heatmap, writes to disk, and returns it for further use.

  The heatmap shows the latency t3 - t2 per node/device for a given experiment.
  The x-axis represents the event index, and the y-axis represents the CUDA
  device. The color of each cell represents the intensity of the latency of the
  corresponding event.

  Args:
    heatmap_data: Data used to plot the heatmap
    output_file: Image file to save the heatmap to
    heatmap_metadata: Metadata used to determine the size of the heatmap.
      Defaults to reasonable sizes for up to 32 nodes.

  Returns:
    A matplotlib figure containing the heatmap.
  """
  # Tweak colormap to focus on slow transfers
  cmap = pl.cm.Reds
  alpha_cmap = cmap(np.arange(cmap.N))
  alpha_cmap[:, -1] = np.linspace(0, 1, cmap.N)
  alpha_cmap = colors.ListedColormap(alpha_cmap)
  fig_width = max(
      heatmap_metadata.min_width,
      min(heatmap_metadata.max_width, len(heatmap_data.x_vals) // 40),
  )
  fig_height = max(
      heatmap_metadata.min_height,
      min(heatmap_metadata.max_height, heatmap_data.max_rank // 8),
  )
  fig, ax = plt.subplots(figsize=(fig_width, fig_height))
  im = ax.imshow(
      heatmap_data.delayed_event_matrix,
      aspect="auto",
      cmap=alpha_cmap,
      vmin=0,
      vmax=heatmap_data.straggler_threshold_ms,
  )
  ax.set_yticks(
      np.arange(len(heatmap_data.y_labels)), labels=heatmap_data.y_labels
  )
  ax.set_xticks(
      np.arange(len(heatmap_data.x_vals)),
      labels=[str(event_idx) for event_idx in heatmap_data.x_vals],
  )
  # Add horizontal lines to group by GPU idx and NIC idx.
  for idx in range(len(heatmap_data.y_labels)):
    if idx % (heatmap_data.max_node_idx + 1) == 0:  # same GPU idx
      ax.axhline(idx - 0.5, ls="-", color="k", linewidth=0.75)
    if idx % (2) == 0:  # helper lines to identify node idx
      ax.axhline(idx - 0.5, ls=":", color="gray", linewidth=0.2)

  # Mark xticks at appropriate intervals to avoid crowding.
  prev_event, prev_marked_event = -10, -10
  x_label_visibility = []
  for idx, event in enumerate(heatmap_data.x_vals):
    is_visible = True
    if event == prev_event + 1:
      is_visible = False
    else:
      ax.axvline(idx, ls="--", color="gray", linewidth=1.0)
    if event >= prev_marked_event + 10:
      is_visible = True
    if is_visible:
      prev_marked_event = event
    prev_event = event
    x_label_visibility.append(is_visible)
  for label, is_visible in zip(ax.get_xticklabels(), x_label_visibility):
    label.set_visible(is_visible)

  plt.setp(
      ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor"
  )
  fig.tight_layout()
  cbar = ax.figure.colorbar(im, ax=ax)
  cbar.ax.set_ylabel("SendRecv duration (ms)", rotation=-90, va="bottom")
  ax.grid(False)
  plt.xlabel("SendRecv event index)")
  plt.ylabel("CUDA Device")
  plt.title(
      "Heatmap of SendRecv latency per node/device for"
      f" {heatmap_data.experiment_name}"
  )

  print(f"Saving heatmap to {output_file}")
  file_extension = output_file.split(".")[-1]
  plt.savefig(output_file, dpi=600, bbox_inches="tight", format=file_extension)
  fig = plt.gcf()
  plt.close(fig)
  return fig
