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

"""Runner for Pipeline Parallelism Benchmark Analysis.

Produces heatmaps for SendRecv latency for each experiment in the data
directory.
"""

import pathlib
from typing import Sequence

from absl import app
from absl import flags

import pp_benchmark_analysis

_DATA_DIR = flags.DEFINE_string(
    "data_dir",
    None,
    "The directory containing the ppbenchmark data.",
    required=True,
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "The directory to write the ppbenchmark analysis results to.",
    required=True,
)

_IMAGE_FILE_NAME = flags.DEFINE_string(
    "image_file_name",
    "straggler_heatmap.svg",
    "The name of the image to use for the heatmap.",
)

_STRAGGLER_THRESHOLD_MS = flags.DEFINE_integer(
    "straggler_threshold_ms",
    None,
    "The threshold for a sendrecv to be considered a straggler.",
    required=True,
)

_INTERESTING_EVENT_OFFSET = flags.DEFINE_integer(
    "interesting_event_offset",
    None,
    "Fwd and bwd offset from a known straggler for marking an event as"
    " interesting",
    required=True,
)

_MINIMUM_CHART_WIDTH = flags.DEFINE_integer(
    "minimum_chart_width",
    16,
    "The minimum width of the chart.",
)

_MAXIMUM_CHART_WIDTH = flags.DEFINE_integer(
    "max_chart_width",
    50,
    "The maximum width of the chart.",
)

_MINIMUM_CHART_HEIGHT = flags.DEFINE_integer(
    "min_chart_height",
    6,
    "The minimum height of the chart.",
)

_MAXIMUM_CHART_HEIGHT = flags.DEFINE_integer(
    "max_chart_height",
    24,
    "The maximum height of the chart.",
)


def main(argv: Sequence[str]):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  data_dir = _DATA_DIR.value
  output_dir = _OUTPUT_DIR.value
  straggler_threshold_ms = _STRAGGLER_THRESHOLD_MS.value
  interesting_event_offset = _INTERESTING_EVENT_OFFSET.value

  experiments = set(
      pathlib.Path(filename).stem.split("-")[1]
      for filename in pathlib.Path(data_dir).glob("*.textproto")
  )
  if not experiments:
    print("No experiments found in dir: %s" % data_dir)
    return

  for experiment in experiments:
    experiment_dir = pathlib.Path(output_dir, experiment)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_data = pp_benchmark_analysis.read_experiment_data(
        data_dir, experiment
    )
    heatmap_data = pp_benchmark_analysis.preprocess_experiment_data(
        benchmark_results=experiment_data,
        experiment=experiment,
        straggler_threshold_ms=straggler_threshold_ms,
        interesting_event_offset=interesting_event_offset,
    )
    if heatmap_data:
      output_file = pathlib.Path(experiment_dir, _IMAGE_FILE_NAME.value)
      _ = pp_benchmark_analysis.plot_straggler_heatmap(
          heatmap_data=heatmap_data,
          output_file=str(output_file),
          heatmap_metadata=pp_benchmark_analysis.HeatmapMetadata(
              min_width=_MINIMUM_CHART_WIDTH.value,
              max_width=_MAXIMUM_CHART_WIDTH.value,
              min_height=_MINIMUM_CHART_HEIGHT.value,
              max_height=_MAXIMUM_CHART_HEIGHT.value,
          ),
      )
    else:
      print(f"No stragglers found for experiment {experiment}.")


if __name__ == "__main__":
  app.run(main)
