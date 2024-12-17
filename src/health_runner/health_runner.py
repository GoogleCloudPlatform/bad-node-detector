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

"""GPU Health Check Daemonset runner.

This application creates a GPU health check daemonset in a Kubernetes cluster,
allows it to run for a specified duration (SLEEP_TIME_MINUTES), and then
terminates it.

Note:
    You must have the necessary permissions to create and delete daemonsets in
    the specified Kubernetes namespace.
"""

from collections.abc import Iterable
import logging
import os
import signal
import time
import uuid

import checker_common
import health_results_pb2
import nccl_runner
from google.protobuf import timestamp_pb2


_SLEEP_TIME_MINUTES = os.environ.get("SLEEP_TIME_MINUTES", "20")
_HELM = os.environ.get("HELM_PATH", "/usr/local/bin/helm")
_KUBECTL = os.environ.get("KUBECTL_PATH", "/app/kubectl")
_GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
_K_NAME_GPU_NODES_IN_CLUSTER_COMMAND = (
    f"{_KUBECTL} get nodes -o jsonpath='{{range"
    ' .items[*]}{.metadata.name}{"\\n"}{end}\' | while read node; do'
    f" {_KUBECTL} get node $node -o"
    " jsonpath='{.status.capacity.nvidia\\.com/gpu}' | grep -q '[0-9]' &&"
    " echo $node; done"
)
_K_NUM_GPU_NODES_IN_CLUSTER_COMMAND = (
    _K_NAME_GPU_NODES_IN_CLUSTER_COMMAND + " | wc -l"
)
logging.root.setLevel(logging.INFO)


def main() -> None:
  # Test for NCCL health check
  health_app = os.environ.get("HEALTH_APP", "").lower()
  # Create Helm releases for each health check
  if health_app:
    logging.info("Running NCCL health check via `HEALTH_APP`")
    run_health_app(health_app)
  else:
    logging.info("Running Helm health check")
    run_health_check()


def run_health_app(health_app: str) -> None:
  """Run the health check."""
  health_results = health_results_pb2.HealthResults(
      created_date_time=timestamp_pb2.Timestamp().GetCurrentTime(),
  )
  if health_app == "nccl":
    logging.info("Running NCCL health check via `HEALTH_APP`")
    nccl_result = nccl_runner.run_nccl_healthcheck()
    health_results.health_results.append(nccl_result)
  else:
    logging.error("Unsupported health app: %s", health_app)
    return

  print(health_results)

  # If GCS_BUCKET_NAME is set, upload the results to GCS.
  if _GCS_BUCKET_NAME:
    checker_common.upload_results_to_gcs(
        bucket_name=_GCS_BUCKET_NAME,
        health_results=health_results,
    )


def ensure_env_variables(required_envs: Iterable[str]) -> None:
  """Ensure necessary environment variables are set."""
  for env in required_envs:
    if env not in os.environ:
      raise ValueError(f"Must set {env}")


# 
def determine_test_iterations() -> int:
  """Determine the number of tests to run.

  This function will calculate the number of tests to deploy for a given test
  run. The behavior is determined by three environment variables:
  * BLAST_MODE_ENABLED - If set to "true", more than one test can be deployed.
  By default this will fill the cluster and run a test on every compatible node.
  * BLAST_MODE_NUM_TESTS_LIMIT - If set to an integer, places a limit on the
  number of tests that will be deployed.
  * NODES_CHECKED_PER_TEST -  This should be set to the number of discrete nodes
  that will be consumed in the YAML_PATH yaml.
  Used to calculate the number of tests to deploy. (Defaults to 1)

  Returns:
  Integer of the number of tests that will be deployed.
  """
  is_blast_mode = str(os.environ.get("BLAST_MODE_ENABLED")).lower() in [
      "true",
      "1",
  ]

  if is_blast_mode:
    logging.info("Running blast mode")

    get_nodes_output = checker_common.run_command(
        _K_NUM_GPU_NODES_IN_CLUSTER_COMMAND
    )
    num_nodes = int(get_nodes_output.stdout)
    nodes_per_test = int(os.environ.get("NODES_CHECKED_PER_TEST", "1"))
    if num_nodes % nodes_per_test != 0:
      logging.warning(
          "Not all nodes can be checked. %d are present on the"
          " cluster. %d will be checked per test."
          " %d node(s) will be unchecked.",
          num_nodes,
          nodes_per_test,
          num_nodes % nodes_per_test,
      )
    max_num_tests = num_nodes // nodes_per_test

    manual_limit_str = os.environ.get("BLAST_MODE_NUM_TESTS_LIMIT")
    if manual_limit_str is not None:
      return min(int(manual_limit_str), max_num_tests)
    else:
      return max_num_tests

  else:
    logging.info("Running single test mode")
    return 1


def is_hc_finished(
    target_count: int = 0,
    timeout_seconds: int = 300,
    check_interval: int = 10,
) -> bool:
  """Waits for pods to reach target count."""

  print(f"Pods to be completed: {target_count=}")

  start_time = time.time()
  while time.time() - start_time < timeout_seconds:
    # TODO: Add a check for the number of pods still running
    time.sleep(check_interval)
    logging.info(
        "(%.2f sec) out of (%d sec)",
        int(time.time() - start_time),
        timeout_seconds,
    )

  message_timeout: str = (
      "Timeout reached after %f seconds."
      " Some pods may still not be in Error or Completed state."
  )
  logging.info(
      message_timeout,
      timeout_seconds,
  )

  return True


def run_health_check() -> None:
  """Run the health check."""

  # PREPARATION
  ensure_env_variables(
      required_envs={
          "DRY_RUN",
          "HELM_CHART",  # Must be defined since can't assume health check type
      },
  )

  # Determine number of tests to run
  num_tests: int = determine_test_iterations()

  cleanup_functions = []

  logging.info("Creating %d tests...", num_tests)
  # This must be defined in the YAML configuration
  helm_chart_path = os.environ.get("HELM_CHART")
  # 
  helm_chart_version = os.environ.get("HELM_CHART_VERSION")
  # 
  helm_install_flags = os.environ.get("HELM_INSTALL_FLAGS")
  # 
  helm_values: dict[str, str] = dict()

  node_names = os.environ.get("HOSTS_CSV", "nil")
  if node_names != "nil":
    node_names = node_names.split(",")
  else:
    node_names = checker_common.run_command(
        _K_NAME_GPU_NODES_IN_CLUSTER_COMMAND
    ).stdout.strip().split("\n")

  num_nodes = os.environ.get("N_NODES", "nil")
  if num_nodes == "nil":
    num_nodes = len(node_names)
  else:
    num_nodes = int(num_nodes)
  node_names_csv = r"\,".join(node_names)

  # Pass Node Names & Number of Nodes to all health checks
  helm_values["health_check.env.HOSTS_CSV"] = f"\"{node_names_csv}\""
  helm_values["health_check.env.N_NODES"] = str(num_nodes)

  # RUN HC
  for i in range(num_tests):
    # If Helm release name is not unique, it will not install the release
    short_guid = str(uuid.uuid4())[:8]
    unique_release_name = os.environ.get(
        "HELM_RELEASE_NAME",
        f"internal-chs-hc-{i}-{short_guid}",
    )
    # Set the job name to a unique value following a specific pattern/format
    # 
    helm_values["job.name"] = f"chs-hc-{i}-{short_guid}"

    cleanup_functions.extend(
        checker_common.create_helm_release(
            helm_path=_HELM,
            release_name=unique_release_name,
            chart=helm_chart_path,
            values=helm_values,
            chart_version=helm_chart_version,
            helm_install_flags=helm_install_flags,
        )
    )
    # TODO - Find a better way to avoid this sleep
    # Sleep to allow time for helm releases to create proper ServiceAccount,
    # ClusterRole, etc.
    time.sleep(1)
    # Count of tests deployed should start at 1 to make it clear
    logging.info("Deployed test %d (%d of %d total)", i, i + 1, num_tests)

  # TODO - Sleep section
  logging.info(
      "Waiting for maximum of %s minutes before cleaning up...",
      _SLEEP_TIME_MINUTES,
  )
  is_hc_finished(
      target_count=num_tests,
      timeout_seconds=int(_SLEEP_TIME_MINUTES) * 60,
      check_interval=10,
  )

  # Cleanup cluster (uninstall helm releases, delete k8s objects, etc.)
  for func in cleanup_functions:
    try:
      func()
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception("Cleanup failed.")


if __name__ == "__main__":
  signal.signal(signal.SIGTERM, checker_common.sigterm_handler)
  main()
