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

"""Runs NCCL health check."""

import collections
from collections.abc import Iterable
import copy
import itertools
import logging
import os
import random
import time
import uuid

import kubernetes.client
from kubernetes.client.api import batch_v1_api

import checker_common
import common_pb2
import health_results_pb2


_SLEEP_TIME_MINUTES = os.environ.get("SLEEP_TIME_MINUTES", "20")
_CHECK_INTERVAL_SECONDS = os.environ.get("CHECK_INTERVAL_SECONDS", "20")

# If set, only run NCCL health check on nodes that have this label set to true
_FILTER_LABEL_NAME = os.environ.get("FILTER_LABEL_NAME", "")
_FILTER_LABEL_VALUE = os.environ.get("FILTER_LABEL_VALUE", "true")

NCCL_PRE_RESULT_KEY = "aiinfra/nccl-healthcheck-pre-result"
NCCL_RESULT_KEY = "aiinfra/nccl-healthcheck-result"


def run_nccl_healthcheck() -> health_results_pb2.HealthResult:
  """Runs NCCL health check and waits for it to complete."""
  print("cleaning up labels")
  labels_to_remove = [
      "aiinfra/nccl-healthcheck-runtime-sec",
      "aiinfra/nccl-healthcheck-result",
      NCCL_PRE_RESULT_KEY,
  ]
  for label in labels_to_remove:
    checker_common.cleanup_labels(label)

  kubernetes.config.load_incluster_config()
  v1 = kubernetes.client.CoreV1Api()
  case = os.environ.get("PAIRING_MODE", "random").lower()
  second_pass_enabled = is_second_pass_enabled()

  node_data: list[dict[str, str]] = get_nodes_data(v1.list_node().items)
  capacity = get_capacity_topology(node_data)
  logging.info("Running %s w/ pairing mode `%s`", "NCCL", case)
  # Default is to use toplogy awareness when pairing nodes
  if case == "intra_rack":
    logging.info("Running %s w/ pairing mode `%s`", "NCCL", case)
    return run_intra_rack_healthcheck(v1, capacity, second_pass_enabled)
  elif case == "inter_rack":
    logging.info("Running %s w/ pairing mode `%s`", "NCCL", case)
    return run_inter_rack_healthcheck(v1, capacity, second_pass_enabled)
  elif case == "inter_cluster":
    logging.info("Running %s w/ pairing mode `%s`", "NCCL", case)
    return run_inter_cluster_healthcheck(v1, capacity, second_pass_enabled)
  elif case == "random":
    logging.info("Running %s w/ pairing mode `%s`", "NCCL", case)
    return run_nccl_random_pair_healthcheck(v1, capacity, second_pass_enabled)
  else:
    logging.info("Unknown health check case: %s", case)
    return run_nccl_random_pair_healthcheck(v1, capacity, second_pass_enabled)


def health_check_with_node_pairs(
    node_pairs: list[tuple[str, str]],
    env_mappings: dict[str, str] | None = None,
    job_name_distinctor: str = "unknown-type",
) -> list[str]:
  """Runs NCCL health check with a list of node pairs.

  Args:
    node_pairs: A list of node pairs to run the health check on.
    env_mappings: A list of additional environment variables to pass to the job.
    job_name_distinctor: A string to distinguish the type of job.

  Returns:
    A list of the nodes that were tested.
  """
  # Create a list of job names for to monitor the jobs.
  job_names = []
  tested_nodes = []
  cleanup_functions = []

  if env_mappings is None:
    env_mappings = {}

  # For the first pass, pair each node
  for node0, node1 in node_pairs:
    env_mappings_copy = copy.deepcopy(env_mappings)
    short_guid = str(uuid.uuid4())[:8]
    unique_job_name = f"chs-hc-{job_name_distinctor}-{short_guid}"
    env_mappings_copy["NODE0"] = node0
    env_mappings_copy["NODE1"] = node1
    env_mappings_copy["SHORT_GUID"] = short_guid

    print(f"Running NCCL test between node {node0} and node {node1}...")
    cleanup_functions.extend(
        checker_common.create_job_k8s(
            job_name=unique_job_name,
            env_mappings=env_mappings_copy,
        )
    )
    # TODO - Find a better way to avoid this sleep
    # Sleep to allow time for helm releases to create proper ServiceAccount,
    # ClusterRole, etc. Otherwise, errors will occur where resources like the
    # ServiceAccount won't exist to create a SSH connection.
    time.sleep(1)

    job_names.append(unique_job_name)
    tested_nodes.append(node0)
    tested_nodes.append(node1)

  print(f"Waiting for {len(job_names)} jobs to complete...")
  job_api = batch_v1_api.BatchV1Api()
  checker_common.wait_till_jobs_complete(
      job_api,
      job_names,
      timeout_seconds=(int(_SLEEP_TIME_MINUTES) * 60),
      check_interval=int(_CHECK_INTERVAL_SECONDS),
  )

  # Cleanup after pods are done (uninstall releases, delete k8s objects, etc.)
  print("All pods are done with first pass")
  for func in cleanup_functions:
    try:
      func()
    except Exception as error:  # pylint: disable=broad-exception-caught
      logging.exception("Cleanup failed (reason: %r).", error)

  return tested_nodes


def run_nccl_random_pair_healthcheck(
    v1: kubernetes.client.CoreV1Api,
    capacity: common_pb2.Capacity,
    second_pass_enabled: bool,
) -> health_results_pb2.HealthResult:
  """Runs NCCL health checks between random nodes and waits for it to complete.

  Args:
    v1: Kubernetes CoreV1Api object.
    capacity: Capacity topology of the cluster.
    second_pass_enabled: Whether second pass is enabled.

  Returns:
    A tuple containing the list of passed nodes and the list of failed nodes.
  """

  health_result = health_results_pb2.HealthResult(name="random_pair")
  nodes = []
  logging.info("Finding all nodes...")
  for cluster in capacity.clusters:
    for rack in cluster.racks:
      for node in rack.nodes:
        nodes.append(node.id)
  logging.info("Found %d nodes", len(nodes))

  # For the first pass, pair each node
  node_pairs = []
  logging.info("Creating node pairs...")
  for i, j in generate_index_pairs(len(nodes)):
    node0 = nodes[i]
    node1 = nodes[j]
    node_pair = (node0, node1)
    node_pairs.append(node_pair)
    logging.info("Paired node %s and node %s", node0, node1)

  tested_nodes = health_check_with_node_pairs(
      node_pairs=node_pairs,
      env_mappings={"SECOND_PASS": "false"},
      job_name_distinctor="nccl-random-pair",
  )

  # Get the passed nodes and other suspect from the first pass.
  node_results = get_nccl_test_results(v1, tested_nodes)
  passed_nodes = node_results.get("pass", list())
  # Consider all other node without "pass" key as suspect
  suspect_nodes: list[tuple[str, str]] = []
  for result_type, nodes in node_results.items():
    if result_type != "pass":
      suspect_nodes.extend((node, result_type) for node in nodes)

  # Label nodes that passed
  for node in passed_nodes:
    checker_common.label_node(
        node, label_key=NCCL_RESULT_KEY, label_value="pass"
    )

  # If no second pass, no failed nodes, or no passed nodes for second pass,
  # then return results.
  if (not second_pass_enabled) or (not suspect_nodes) or (not passed_nodes):
    logging.info("Second pass will not run")
    # Label suspect nodes based on their given result type
    for node, result_type in suspect_nodes:
      checker_common.label_node(
          node, label_key=NCCL_RESULT_KEY, label_value=result_type
      )
    failed_nodes = node_results.get("fail", list())
    suspect_nodes_no_fails: list[str] = [
        node for node, result_type in suspect_nodes if result_type != "fail"
    ]
    print(f"Found suspect nodes: {suspect_nodes_no_fails}")
    print(f"Found failed nodes: {failed_nodes}")
    print(f"Found passed nodes: {passed_nodes}")
    health_result.nccl_health_result.CopyFrom(
        generate_nccl_health_results(passed_nodes, failed_nodes)
    )
    return health_result

  print(f"Running second pass for {len(suspect_nodes)} nodes...")

  second_pass_node_pairs = []
  # For second pass, pair each suspect node with a randomly selected passed node
  # If there are more suspect nodes than passed nodes, passed nodes will be
  # cycled through and therefore pair with more than one suspect node.
  passed_nodes_list = list(passed_nodes)
  # Get just the names from suspect nodes
  suspect_nodes_list = [node for (node, _) in suspect_nodes]
  random.shuffle(passed_nodes_list)
  for suspect_node, good_node in zip(
      suspect_nodes_list, itertools.cycle(passed_nodes_list)
  ):
    node_pair = (suspect_node, good_node)
    second_pass_node_pairs.append(node_pair)
    print(
        f"Will run NCCL test between good node {good_node} and suspect node"
        f" {suspect_node}"
    )

  tested_nodes_second_pass = health_check_with_node_pairs(
      node_pairs=second_pass_node_pairs,
      env_mappings={"SECOND_PASS": "true", "HEALTH_VALIDITY_HOURS": "0"},
      job_name_distinctor="nccl-2nd-pass",
  )
  print(f"Second pass completed for {len(tested_nodes_second_pass)} nodes")

  # Only care about the results from the previous failed nodes
  node_results_second_pass = get_nccl_test_results(v1, suspect_nodes_list)
  passed_nodes_second_pass = node_results_second_pass.get("pass", list())
  # Consider all other node without "pass" key as suspect
  suspect_nodes_second_pass: list[tuple[str, str]] = []
  for result_type, nodes in node_results_second_pass.items():
    if result_type != "pass":
      suspect_nodes_second_pass.extend(
          (node, result_type)
          for node in nodes
          if node not in passed_nodes_second_pass
      )
  # Get just the names from suspect nodes for second pass
  suspect_nodes_second_pass_list = [
      node for (node, _) in suspect_nodes_second_pass
  ]

  # Label nodes that passed second pass
  for node in passed_nodes_second_pass:
    checker_common.label_node(
        node, label_key=NCCL_RESULT_KEY, label_value="pass"
    )
  # Since final test, we can use this result type as final result
  for node, result_type in suspect_nodes_second_pass:
    checker_common.label_node(
        node, label_key=NCCL_RESULT_KEY, label_value=result_type
    )
  passed_nodes, failed_nodes = determine_failed_components(
      passed_nodes,
      suspect_nodes_list,
      passed_nodes_second_pass,
      suspect_nodes_second_pass_list,
  )

  print(f"found failed/suspect nodes: {failed_nodes}")
  print(f"found passed nodes: {passed_nodes}")
  health_result.nccl_health_result.CopyFrom(
      generate_nccl_health_results(passed_nodes, failed_nodes)
  )
  return health_result


def run_intra_rack_healthcheck(
    v1: kubernetes.client.CoreV1Api,
    capacity: common_pb2.Capacity,
    second_pass_enabled: bool,
) -> health_results_pb2.HealthResult:
  """Checks the racks communication by running nccl tests between each node in the same rack."""
  health_result = health_results_pb2.HealthResult(
      name="intra_rack",
  )
  # Create a dictionary of rack to nodes for easier lookup.
  rack_to_nodes = {}
  logging.info("Finding all racks...")
  for cluster in capacity.clusters:
    for rack in cluster.racks:
      rack_to_nodes[rack.id] = [node.id for node in rack.nodes]
  logging.info("Found %d racks", len(rack_to_nodes))

  # Contains all node pairs created by rack
  node_pairs = []

  # For the first pass, pair each node in the rack
  for rack, nodes_in_rack in rack_to_nodes.items():
    logging.info("%d nodes in rack %s", len(nodes_in_rack), rack)
    if len(nodes_in_rack) < 2:
      print(f"Skipping rack {rack} with less than 2 nodes.")
      continue
    # Add the specific node pairs to the overall list
    for i, j in generate_index_pairs(len(nodes_in_rack)):
      node0 = nodes_in_rack[i]
      node1 = nodes_in_rack[j]
      node_pair = (node0, node1)
      node_pairs.append(node_pair)
      print(
          f"Will run NCCL test between good node {node0} (Rack:"
          f" {rack}) and failed node {node1} (Rack:"
          f" {rack})"
      )

  tested_nodes = health_check_with_node_pairs(
      node_pairs=node_pairs,
      env_mappings={"SECOND_PASS": "false"},
      job_name_distinctor="nccl-intra-rack",
  )

  # Get the suspect and passed nodes from the first pass.
  node_results = get_nccl_test_results(v1, tested_nodes)
  passed_nodes = node_results.get("pass", list())
  # Consider all other node without "pass" key as suspect
  suspect_nodes: list[tuple[str, str]] = []
  for result_type, nodes in node_results.items():
    if result_type != "pass":
      suspect_nodes.extend((node, result_type) for node in nodes)
  suspect_nodes_list = [node for (node, _) in suspect_nodes]

  # Label nodes that passed
  for node in passed_nodes:
    checker_common.label_node(
        node, label_key=NCCL_RESULT_KEY, label_value="pass"
    )

  # If no second pass, no failed nodes, or no passed nodes for second pass,
  # then return results.
  if (not second_pass_enabled) or (not suspect_nodes) or (not passed_nodes):
    logging.info("Second pass will not run")
    # Label suspect nodes based on their given result type
    for node, result_type in suspect_nodes:
      checker_common.label_node(
          node, label_key=NCCL_RESULT_KEY, label_value=result_type
      )
    failed_nodes = node_results.get("fail", list())
    suspect_nodes_no_fails: list[str] = [
        node for node, result_type in suspect_nodes if result_type != "fail"
    ]
    print(f"Found suspect nodes: {suspect_nodes_no_fails}")
    print(f"Found failed nodes: {failed_nodes}")
    print(f"Found passed nodes: {passed_nodes}")
    health_result.nccl_health_result.CopyFrom(
        generate_nccl_health_results(passed_nodes, failed_nodes)
    )
    return health_result

  # For the second pass, pair each suspect node with a passed node in the same
  # rack.
  print(f"Running second pass for {len(suspect_nodes)} nodes...")
  second_pass_node_pairs = []
  all_suspect_nodes = []
  for rack, nodes in rack_to_nodes.items():
    passed_nodes_in_rack = []
    suspect_nodes_in_rack = []
    # Determine which nodes in the rack passed and which suspect.
    for node in nodes:
      if node in passed_nodes:
        passed_nodes_in_rack.append(node)
      elif node in suspect_nodes_list:
        suspect_nodes_in_rack.append(node)

    if not passed_nodes_in_rack:
      print(f"No passed nodes in rack: {rack}")
      continue

    # Shuffle the passed nodes in the rack & then cycle through them to pair
    # with each suspect node exhaustively. Repeats are fine but not ideal.
    random.shuffle(passed_nodes_in_rack)
    for suspect_node, healthy_node in zip(
        suspect_nodes_in_rack, itertools.cycle(passed_nodes_in_rack)
    ):
      node_pair = (suspect_node, healthy_node)
      second_pass_node_pairs.append(node_pair)
      print(
          f"Will run NCCL test between good node {healthy_node} (Rack:"
          f" {rack}) and suspect node {suspect_node} (Rack:"
          f" {rack})"
      )
    # Track all suspect nodes from the first pass (should have no repeats)
    all_suspect_nodes.extend(suspect_nodes_in_rack)

  tested_nodes_second_pass = health_check_with_node_pairs(
      node_pairs=second_pass_node_pairs,
      env_mappings={"SECOND_PASS": "true"},
      job_name_distinctor="nccl-intra-rack-second-pass",
  )
  print(f"Second pass completed for {len(tested_nodes_second_pass)} nodes")

  # Only care about the results from the previous failed nodes
  node_results_second_pass = get_nccl_test_results(v1, all_suspect_nodes)
  passed_nodes_second_pass = node_results_second_pass.get("pass", list())
  # Consider all other node without "pass" key as suspect
  suspect_nodes_second_pass: list[tuple[str, str]] = []
  for result_type, nodes in node_results_second_pass.items():
    if result_type != "pass":
      suspect_nodes_second_pass.extend(
          (node, result_type)
          for node in nodes
          if node not in passed_nodes_second_pass
      )

  # Label nodes that passed second pass
  for node in passed_nodes_second_pass:
    checker_common.label_node(
        node, label_key=NCCL_RESULT_KEY, label_value="pass"
    )
  # Since final test, we can use this result type as final result
  for node, result_type in suspect_nodes_second_pass:
    checker_common.label_node(
        node, label_key=NCCL_RESULT_KEY, label_value=result_type
    )
  # Get just the names from suspect nodes for second pass
  suspect_nodes_second_pass_list = [
      node for (node, _) in suspect_nodes_second_pass
  ]
  passed_nodes, failed_nodes = determine_failed_components(
      passed_nodes,
      suspect_nodes_list,
      passed_nodes_second_pass,
      suspect_nodes_second_pass_list,
  )

  print(f"found failed/suspect nodes: {failed_nodes}")
  print(f"found passed nodes: {passed_nodes}")
  health_result.nccl_health_result.CopyFrom(
      generate_nccl_health_results(passed_nodes, failed_nodes)
  )
  return health_result


def run_inter_rack_healthcheck(
    v1: kubernetes.client.CoreV1Api,
    capacity: common_pb2.Capacity,
    second_pass_enabled: bool,
) -> health_results_pb2.HealthResult:
  """Checks inter-rack communication by running nccl tests between nodes in different racks (The racks will be in the same cluster)."""
  health_result = health_results_pb2.HealthResult(
      name="inter_rack",
  )
  cluster_to_racks = {}
  rack_to_nodes = {}
  node_pairs = []

  # Create a dictionary of cluster to racks and a dictionary of rack to nodes
  # to make it easier to look up info.
  logging.info("Finding all racks...")
  for cluster in capacity.clusters:
    cluster_to_racks[cluster.id] = []
    for rack in cluster.racks:
      cluster_to_racks[cluster.id].append(rack.id)
      rack_to_nodes[rack.id] = [node.id for node in rack.nodes]
  logging.info("Found %d racks", len(rack_to_nodes))

  for _, racks in cluster_to_racks.items():
    # Pair each rack with another rack in the same cluster
    rack_pairs = generate_index_pairs(len(racks))
    for i, j in rack_pairs:
      rack0 = racks[i]
      rack1 = racks[j]
      rack0_nodes = rack_to_nodes[rack0]
      logging.info("%d nodes in rack %s", len(rack0_nodes), rack0)
      rack1_nodes = rack_to_nodes[rack1]
      logging.info("%d nodes in rack %s", len(rack1_nodes), rack1)
      if not rack0_nodes or not rack1_nodes:
        raise ValueError(f"Rack: {racks[i]} or {racks[j]} has no nodes.")
      # Randomly select a node from each rack
      node0 = random.choice(rack0_nodes)
      node1 = random.choice(rack1_nodes)
      node_pair = (node0, node1)
      node_pairs.append(node_pair)
      print(
          f"Will run NCCL test between node {node0} (Rack:"
          f" {rack0}) and node {node1} (Rack:"
          f" {rack1})"
      )

  tested_nodes = health_check_with_node_pairs(
      node_pairs=node_pairs,
      env_mappings={"SECOND_PASS": "false"},
      job_name_distinctor="nccl-inter-rack",
  )

  # Get the passed nodes and other suspect from the first pass.
  node_results = get_nccl_test_results(v1, tested_nodes)
  passed_nodes = node_results.get("pass", list())
  # Consider all other node without "pass" key as suspect
  suspect_nodes: list[tuple[str, str]] = []
  for result_type, nodes in node_results.items():
    if result_type != "pass":
      suspect_nodes.extend((node, result_type) for node in nodes)

  # Label nodes that passed
  for node in passed_nodes:
    checker_common.label_node(
        node, label_key=NCCL_RESULT_KEY, label_value="pass"
    )

  passed_racks = []
  failed_racks = []

  # Determine which racks passed and which failed by checking if any of the
  # nodes in the rack passed.
  for rack, nodes in rack_to_nodes.items():
    if any(node in passed_nodes for node in nodes):
      passed_racks.append(rack)
    else:
      failed_racks.append(rack)

  if (not second_pass_enabled) or (not failed_racks) or (not passed_nodes):
    logging.info("Second pass will not run")
    for node, result_type in suspect_nodes:
      checker_common.label_node(
          node, label_key=NCCL_RESULT_KEY, label_value=result_type
      )
    print(f"Found failed racks: {failed_racks}")
    print(f"Found passed racks: {passed_racks}")
    health_result.nccl_health_result.CopyFrom(
        generate_nccl_health_results(passed_racks, failed_racks)
    )
    return health_result

  # If second pass is enabled, we will run the test between the passed and
  # failed racks in the same cluster.
  print(f"Running second pass for {len(failed_racks)} racks...")
  second_pass_node_pairs = []
  all_failed_nodes = []
  for cluster in capacity.clusters:
    for failed_rack in cluster.racks:
      if failed_rack.id not in failed_racks:
        continue

      # Get the list of racks in the same cluster as the failed rack.
      potential_racks = cluster_to_racks[cluster.id]

      # Choose a random healthy rack from the same cluster
      healthy_rack = ""
      for potential_rack in potential_racks:
        if potential_rack in passed_racks:
          healthy_rack = potential_rack
          break

      if not healthy_rack:
        print(f"No healthy rack found for rack: {failed_rack.id}")
        continue

      # Choose a random node from the healthy and failed rack
      failed_nodes = rack_to_nodes[failed_rack.id]
      healthy_nodes = rack_to_nodes[healthy_rack]

      healthy_node = random.choice(healthy_nodes)
      failed_node = random.choice(failed_nodes)
      all_failed_nodes.append(failed_node)
      node_pair = (healthy_node, failed_node)
      second_pass_node_pairs.append(node_pair)

      print(
          f"Will run NCCL test between good node {healthy_node} (Rack:"
          f" {healthy_rack}) and failed node {failed_node} (Rack:"
          f" {failed_rack})"
      )

  tested_nodes = health_check_with_node_pairs(
      node_pairs=second_pass_node_pairs,
      env_mappings={"SECOND_PASS": "true"},
      job_name_distinctor="nccl-inter-rack-second-pass",
  )

  # Get the results of the second pass and combine with the first pass results
  node_results_second_pass = get_nccl_test_results(v1, tested_nodes)
  passed_nodes_second_pass = node_results_second_pass.get("pass", list())
  # Consider all other node without "pass" key as suspect
  suspect_nodes_second_pass: list[tuple[str, str]] = []
  for result_type, nodes in node_results_second_pass.items():
    if result_type != "pass":
      suspect_nodes_second_pass.extend(
          (node, result_type)
          for node in nodes
          if node not in passed_nodes_second_pass
      )

  # Label nodes that passed second pass
  for node in passed_nodes_second_pass:
    checker_common.label_node(
        node, label_key=NCCL_RESULT_KEY, label_value="pass"
    )
  # Since final test, we can use this result type as final result
  for node, result_type in suspect_nodes_second_pass:
    checker_common.label_node(
        node, label_key=NCCL_RESULT_KEY, label_value=result_type
    )

  second_passed_racks = []
  second_failed_racks = []

  # Determine which racks passed and which failed by checking if any of the
  # nodes in the rack passed.
  for rack, nodes in rack_to_nodes.items():
    if any(node in passed_nodes_second_pass for node in nodes):
      second_passed_racks.append(rack)
    else:
      second_failed_racks.append(rack)

  # Combine the results of the first and second pass.
  passed_racks, failed_racks = determine_failed_components(
      passed_racks, failed_racks, second_passed_racks, second_failed_racks
  )

  print(f"After second pass - Failed racks: {failed_racks}")
  print(f"After second pass - Passed racks: {passed_racks}")
  health_result.nccl_health_result.CopyFrom(
      generate_nccl_health_results(passed_racks, failed_racks)
  )

  return health_result


def run_inter_cluster_healthcheck(
    v1: kubernetes.client.CoreV1Api,
    capacity: common_pb2.Capacity,
    second_pass_enabled: bool,
) -> health_results_pb2.HealthResult:
  """Checks the inter-cluster communication by running nccl tests between nodes in different clusters."""
  health_result = health_results_pb2.HealthResult(
      name="inter_cluster",
  )
  cluster_to_nodes = {}
  for cluster in capacity.clusters:
    for rack in cluster.racks:
      if cluster.id not in cluster_to_nodes:
        cluster_to_nodes[cluster.id] = [node.id for node in rack.nodes]
      else:
        cluster_to_nodes[cluster.id].extend([node.id for node in rack.nodes])

  node_pairs = []

  # Pair each cluster with another cluster
  cluster_pairs = generate_index_pairs(len(capacity.clusters))
  for i, j in cluster_pairs:
    cluster0_nodes = cluster_to_nodes[capacity.clusters[i].id]
    cluster1_nodes = cluster_to_nodes[capacity.clusters[j].id]
    if not cluster0_nodes or not cluster1_nodes:
      raise ValueError(
          f"Cluster {capacity.clusters[i].id} or {capacity.clusters[j].id} has"
          " no racks."
      )

    # Randomly select a node from each cluster
    node0 = random.choice(cluster0_nodes)
    node1 = random.choice(cluster1_nodes)
    node_pair = (node0, node1)
    node_pairs.append(node_pair)

    print(
        f"Will run NCCL test between node {node0} (Cluster:"
        f" {capacity.clusters[i].id}) and node"
        f" {node1} (Cluster: {capacity.clusters[j].id})"
    )

  tested_nodes = health_check_with_node_pairs(
      node_pairs=node_pairs,
      env_mappings={"SECOND_PASS": "false"},
      job_name_distinctor="nccl-inter-cluster",
  )

  # Check for failures and attempt second pass
  node_results = get_nccl_test_results(v1, tested_nodes)
  passed_nodes = node_results.get("pass", list())
  # Consider all other node without "pass" key as suspect
  suspect_nodes: list[tuple[str, str]] = []
  for result_type, nodes in node_results.items():
    if result_type != "pass":
      suspect_nodes.extend((node, result_type) for node in nodes)

  # Label nodes that passed
  for node in passed_nodes:
    checker_common.label_node(
        node, label_key=NCCL_RESULT_KEY, label_value="pass"
    )

  passed_clusters = []
  failed_clusters = []
  for cluster in capacity.clusters:
    if any(node in passed_nodes for node in cluster_to_nodes[cluster.id]):
      passed_clusters.append(cluster.id)
    else:
      failed_clusters.append(cluster.id)

  if not second_pass_enabled or not failed_clusters or not passed_nodes:
    logging.info("Second pass will not run")
    # Label suspect nodes based on their given result type
    for node, result_type in suspect_nodes:
      checker_common.label_node(
          node, label_key=NCCL_RESULT_KEY, label_value=result_type
      )
    print(f"Found failed clusters: {failed_clusters}")
    print(f"Found passed clusters: {passed_clusters}")
    health_result.nccl_health_result.CopyFrom(
        generate_nccl_health_results(passed_clusters, failed_clusters)
    )
    return health_result

  print(f"Running second pass for {len(failed_clusters)} clusters...")
  second_pass_node_pairs = []

  # Loop through the failed clusters and pair it with a random healthy cluster
  all_suspect_nodes = []
  for failed_cluster in failed_clusters:
    healthy_cluster = random.choice(passed_clusters)

    suspect_nodes_in_cluster = cluster_to_nodes[failed_cluster]
    healthy_nodes = cluster_to_nodes[healthy_cluster]

    # Choose a random healthy node from a healthy cluster
    healthy_node = random.choice(healthy_nodes)
    suspect_node = random.choice(suspect_nodes_in_cluster)
    all_suspect_nodes.append(suspect_node)
    node_pair = (suspect_node, healthy_node)
    second_pass_node_pairs.append(node_pair)

    print(
        f"Will run NCCL test between good node {healthy_node} (Cluster:"
        f" {healthy_cluster}) and suspect node {suspect_node} (Cluster:"
        f" {failed_cluster})"
    )

  tested_nodes = health_check_with_node_pairs(
      node_pairs=second_pass_node_pairs,
      env_mappings={"SECOND_PASS": "true"},
      job_name_distinctor="nccl-inter-cluster-second-pass",
  )

  node_results_second_pass = get_nccl_test_results(v1, tested_nodes)
  passed_nodes_second_pass = node_results_second_pass.get("pass", list())
  # Consider all other node without "pass" key as suspect
  suspect_nodes_second_pass: list[tuple[str, str]] = []
  for result_type, nodes in node_results_second_pass.items():
    if result_type != "pass":
      suspect_nodes_second_pass.extend(
          (node, result_type)
          for node in nodes
          if node not in passed_nodes_second_pass
      )

  # Label nodes that originally failed the second pass
  for node in all_suspect_nodes:
    if node in passed_nodes_second_pass:
      checker_common.label_node(
          node, label_key=NCCL_RESULT_KEY, label_value="pass"
      )
  # Since final test, we can use this result type as final result
  for node, result_type in suspect_nodes_second_pass:
    if node in all_suspect_nodes:
      checker_common.label_node(
          node, label_key=NCCL_RESULT_KEY, label_value=result_type
      )

  second_passed_clusters = []
  second_failed_clusters = []
  for cluster in capacity.clusters:
    if any(
        (node in passed_nodes_second_pass)
        for node in cluster_to_nodes[cluster.id]
    ):
      second_passed_clusters.append(cluster.id)
    else:
      second_failed_clusters.append(cluster.id)

  # Combine the results of the first and second pass.
  passed_clusters, failed_clusters = determine_failed_components(
      passed_clusters,
      failed_clusters,
      second_passed_clusters,
      second_failed_clusters,
  )

  print(f"After second pass - Failed racks: {failed_clusters}")
  print(f"After second pass - Passed racks: {passed_clusters}")
  health_result.nccl_health_result.CopyFrom(
      generate_nccl_health_results(passed_clusters, failed_clusters)
  )

  return health_result


# 
def determine_failed_components(
    first_pass_passed: list[str],
    first_pass_failed: list[str],
    second_pass_passed: list[str],
    second_pass_failed: list[str],
) -> tuple[list[str], list[str]]:
  """Determines the final list of failed and passed nodes."""
  passed_set = set(first_pass_passed)
  failed_set = set(second_pass_failed)

  # Add any node that passed the second pass to the passed set.
  for node in second_pass_passed:
    if node not in passed_set:
      passed_set.add(node)

  # Remove any node that passed the first pass but failed the second pass.
  # Nodes can not move from passed to failed.
  first_pass_failed_set = set(first_pass_failed)
  for node in second_pass_failed:
    if node not in first_pass_failed_set and node in failed_set:
      failed_set.remove(node)

  # Add any node that failed the first pass and is not in the passed set.
  # This can happen if the node was not tested in the second pass.
  for node in first_pass_failed_set:
    if node not in passed_set:
      failed_set.add(node)

  return list(passed_set), list(failed_set)


def get_nccl_test_results(
    v1: kubernetes.client.CoreV1Api,
    nodes: list[str],
) -> dict[str, list[str]]:
  """Returns the NCCL test results for the given nodes.

  Args:
    v1: The Kubernetes API client.
    nodes: The list of nodes to get the results for.

  Returns:
    A dictionary of node results with keys as result types and values as lists
    of node names.
  """

  node_results: dict[str, list[str]] = collections.defaultdict(list)

  tested_nodes = set(nodes)

  nodes = v1.list_node()
  for node in nodes.items:
    if node.metadata.name not in tested_nodes:
      continue
    pre_result = node.metadata.labels.get(NCCL_PRE_RESULT_KEY)
    node_name = node.metadata.name
    logging.info(
        "Node %s has result: %s.",
        node_name,
        pre_result,
    )
    # If bandwidth is > threshold, then it passed. Otherwise a fail
    match pre_result:
      case "pass":
        node_results["pass"].append(node_name)
        logging.info(
            "Node %s will be considered 'pass'.",
            node_name,
        )
      case None:
        node_results["timeout"].append(node_name)
        logging.info(
            "Node %s will be considered 'timeout'.",
            node_name,
        )
      case "crash":
        node_results["crash"].append(node_name)
        logging.info(
            "Node %s will be considered 'crash'.",
            node_name,
        )
      case _:
        node_results["fail"].append(node_name)
        logging.info(
            "Node %s will be considered failed.",
            node_name,
        )
  return node_results


def generate_index_pairs(length: int) -> list[tuple[int, int]]:
  """Returns random pairs of indices with no repeated items."""
  if length < 2:
    return []

  indices = list(range(length))
  random.shuffle(indices)
  pairs = []

  while len(indices) > 1:
    index1 = indices.pop()
    index2 = indices.pop()
    pairs.append((index1, index2))

  # If there's an odd number of indices, pair the last one randomly
  if indices:
    last_index = indices.pop()
    random_partner = random.randint(0, length - 1)
    # Ensure the index doesn't pair with itself
    while random_partner == last_index:
      random_partner = random.randint(0, length - 1)
    pairs.append((last_index, random_partner))

  return pairs


def get_node_data_v1(
    gpu_nodes: list[kubernetes.client.models.V1Node],
) -> list[dict[str, str]]:
  """Returns a list of node data from the given list of nodes.

  Works on topology information provided by the v1 instance API.

  Args:
    gpu_nodes: List of GPU nodes.

  Returns:
    List of node data.
  """
  nodes = []
  for gpu_node in gpu_nodes:
    if "topology.gke.io/cluster" not in gpu_node.metadata.labels:
      continue

    node = {}
    node["cluster"] = gpu_node.metadata.labels.get(
        "topology.gke.io/cluster", "unknown"
    )
    node["rack"] = gpu_node.metadata.labels.get(
        "topology.gke.io/rack", "unknown"
    )
    node["host"] = gpu_node.metadata.labels.get(
        "topology.gke.io/host", "unknown"
    )
    node["node_id"] = gpu_node.metadata.name
    nodes.append(node)
  return nodes


def get_node_data_v2(
    gpu_nodes: Iterable[kubernetes.client.models.V1Node],
) -> list[dict[str, str]]:
  """Returns a list of node data from the given list of nodes.

  Works on topology information provided by the v2 instance API.

  Args:
    gpu_nodes: List of GPU nodes.

  Returns:
    List of node data.
  """
  nodes = []
  for gpu_node in gpu_nodes:
    node = {}
    node["cluster"] = gpu_node.metadata.labels.get(
        "cloud.google.com/gce-topology-block", "unknown"
    )
    node["rack"] = gpu_node.metadata.labels.get(
        "cloud.google.com/gce-topology-subblock", "unknown"
    )
    node["host"] = gpu_node.metadata.labels.get(
        "cloud.google.com/gce-topology-host", "unknown"
    )
    node["node_id"] = gpu_node.metadata.name
    nodes.append(node)
  return nodes


def get_nodes_data(
    kube_nodes: list[kubernetes.client.models.V1Node],
) -> list[dict[str, str]]:
  """Returns a list of node data from the given list of nodes & conditions.

  Args:
    kube_nodes: List of nodes.

  Returns:
    List of node data.
  """
  # Filter set of nodes before getting data
  gpu_nodes = []
  for node in kube_nodes:
    # Must have a GPU label
    if not has_gpu_resources(node):
      continue

    # If filter label is specified, then the node must have the label and the
    # value must match
    if _FILTER_LABEL_NAME and (
        _FILTER_LABEL_NAME not in node.metadata.labels
        or node.metadata.labels[_FILTER_LABEL_NAME] != _FILTER_LABEL_VALUE
    ):
      continue

    gpu_nodes.append(node)

  # If no nodes are found then return an empty list
  if not gpu_nodes:
    return []

  # Use the first node to determine the topology version.
  if "topology.gke.io/cluster" in gpu_nodes[0].metadata.labels:
    return get_node_data_v1(gpu_nodes)
  elif "cloud.google.com/gce-topology-host" in gpu_nodes[0].metadata.labels:
    return get_node_data_v2(gpu_nodes)
  else:
    # If no topology labels are found then all nodes will be grouped under a
    # single cluster and rack with the id "unknown"
    print("No topology labels found.")
    return get_node_data_v2(gpu_nodes)


def has_gpu_resources(node):
  """Check if the node has GPU resources in capacity or allocatable.

  Args:
      node (kubernetes.client.V1Node): Kubernetes node object

  Returns:
      bool: True if node has GPUs, False otherwise
  """
  # Get the node's status
  node_status = node.status
  # Check both capacity and allocatable for GPU resources
  if node_status and node_status.allocatable:
    gpu_key = "nvidia.com/gpu"  # Standard NVIDIA GPU label

    if gpu_key in node_status.allocatable:
      # Convert to int and check if > 0
      try:
        gpu_count = int(node_status.allocatable.get(gpu_key, 0))
        if gpu_count > 0:
          return True
      except (TypeError, ValueError):
        pass

  return False


def get_capacity_topology(
    nodes: list[dict[str, str]],
) -> common_pb2.Capacity:
  """Creates a capacity topology from the list of nodes."""

  capacity = common_pb2.Capacity()
  cluster_dict = dict()
  rack_dict = dict()

  # If nodes don't have all topology labels then all nodes will be grouped under
  # a single cluster and rack with the id "unknown"
  for node_data in nodes:
    cluster_id = node_data["cluster"]
    rack_id = node_data["rack"]
    node_id = node_data["node_id"]
    host_id = node_data["host"]

    if cluster_id not in cluster_dict:
      cluster = capacity.clusters.add()
      cluster.id = cluster_id
      cluster_dict[cluster_id] = cluster
    cluster = cluster_dict[cluster_id]

    if rack_id not in rack_dict:
      rack = cluster.racks.add()
      rack.id = rack_id
      rack_dict[rack_id] = rack

    rack = rack_dict[rack_id]
    rack.nodes.append(common_pb2.Node(id=node_id, host=host_id))

  return capacity


def generate_nccl_health_results(
    passed_objects: Iterable[str],
    failed_objects: Iterable[str],
) -> health_results_pb2.NCCLHealthResultList:
  """Generates a list of NCCLHealthResult protos for the given nodes."""
  health_results = health_results_pb2.NCCLHealthResultList()
  # Sort the objects to ensure the results are deterministic
  objects = list(passed_objects) + list(failed_objects)
  objects.sort()

  passed_objects = set(passed_objects)
  for obj in objects:
    if obj in passed_objects:
      status = health_results_pb2.Status.PASS
    else:
      status = health_results_pb2.Status.FAIL
    health_results.nccl_health_results.append(
        health_results_pb2.NCCLHealthResult(id=obj, status=status)
    )
  return health_results


def is_second_pass_enabled() -> bool:
  return os.environ.get("SECOND_PASS_ENABLED", "true").lower() == "true"
