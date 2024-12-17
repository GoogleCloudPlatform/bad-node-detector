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

#!/bin/bash

# Set NCCL Variables
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-4}"
export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-4}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_PROTO="${NCCL_PROTO:-Simple}"
export NCCL_ALGO="${NCCL_ALGO:-RING}"

export NCCL_P2P_PXN_LEVEL="${NCCL_P2P_PXN_LEVEL:-0}"
export NCCL_P2P_PCI_CHUNKSIZE="${NCCL_P2P_PCI_CHUNKSIZE:-524288}"
export NCCL_P2P_NVL_CHUNKSIZE="${NCCL_P2P_NVL_CHUNKSIZE:-1048576}"
export NCCL_P2P_NET_CHUNKSIZE="${NCCL_P2P_NET_CHUNKSIZE:-524288}"

export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PIX}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-8}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
export NCCL_GRAPH_MIXING_SUPPORT="${NCCL_GRAPH_MIXING_SUPPORT:-0}"
export NCCL_DYNAMIC_CHUNK_SIZE="${NCCL_DYNAMIC_CHUNK_SIZE:-524288}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
export NCCL_CHECK_POINTERS="${NCCL_CHECK_POINTERS:-0}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-8388608}"
export NCCL_DEBUG="${NCCL_DEBUG:-DEBUG}"  # set "INFO" when debugging or it may affect NCCL perf
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-ENV}" # set "INIT,GRAPH,ENV,TUNING" when debugging

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export ACCELERATORS_PER_POD=8

export NCCL_FASTRAK_USE_SNAP=1
export NCCL_FASTRAK_USE_LLCM=1
export NCCL_FASTRAK_CTRL_DEV="${NCCL_FASTRAK_CTRL_DEV:-eth0}"
export NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING="${NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING:-0}"
export NCCL_FASTRAK_IFNAME="${NCCL_FASTRAK_IFNAME:-eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8}"
export NCCL_FASTRAK_NUM_FLOWS="${NCCL_FASTRAK_NUM_FLOWS:-8}"
export NCCL_FASTRAK_FLOWS_PER_GROUP="${NCCL_FASTRAK_FLOWS_PER_GROUP:-2}"
export FASTRAK_MIN_NCHANNELS="${FASTRAK_MIN_NCHANNELS:-16}"
export FASTRAK_MAX_NCHANNELS="${FASTRAK_MAX_NCHANNELS:-16}"

export NCCL_DEBUG_FILE="/tmp/log/nccl-%h-%p.log"
export NCCL_TOPO_DUMP_FILE="/tmp/log/nccl_topo.txt"
export NCCL_GRAPH_DUMP_FILE="/tmp/log/nccl_graph.txt"

export NCCL_SHIMNET_SHIM_LAYERS="${NCCL_SHIMNET_SHIM_LAYERS:-UNUSED}"
export NCCL_TUNER_PLUGIN="${NCCL_TUNER_PLUGIN:-UNUSED}"

export LD_LIBRARY_PATH="/usr/local/fastrak_exec/lib64:${LD_LIBRARY_PATH}"

export N_BATCH=${N_BATCH:-100}
export N_MICROBATCH=${N_MICROBATCH:-1000}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}

HOST_NAME="$(hostname)"
export HOST_NAME

# 
# Mount GCS bucket
chmod 664 /gcs
export GCS_BUCKET="${GCS_BUCKET:-sd_mvp}"
gcsfuse --implicit-dirs "$GCS_BUCKET" /gcs

# Launch ssh
service ssh restart

NPROCESS=$(($N_NODES * $N_GPUS_PER_NODE))

if [[ -n "${HOSTS_CSV}" ]]; then
    echo "The HOSTS_CSV is: $HOSTS_CSV"
    HOSTS_CSV_WO_SLOTS=$(echo "${HOSTS_CSV}")
    echo "The HOSTS_CSV file is: $HOSTS_CSV_WO_SLOTS"
    IFS=',' read -ra ADDR <<< "$HOSTS_CSV_WO_SLOTS"
    MPI_HOSTS=""
    for i in "${ADDR[@]}"; do
        MPI_HOSTS+="${i}:8,"
    done
    MPI_HOSTS=${MPI_HOSTS%?}
fi

mkdir /usr/local/fastrak_exec
mount --bind /usr/local/fastrak_exec /usr/local/fastrak_exec
mount -o remount,exec /usr/local/fastrak_exec
cp -r /usr/local/fastrak/lib64 /usr/local/fastrak_exec

MPI_ENV_VAR="LD_LIBRARY_PATH=/usr/local/fastrak_exec/lib64:${LD_LIBRARY_PATH}"

function on_script_completion {
    # Note: This semaphore is tracked by rxdm daemon container in gke
    mkdir -p /usr/share/nemo
    touch /usr/share/nemo/workload_terminated
}

rm -f /usr/share/nemo/workload_terminated
trap on_script_completion EXIT


# Update iptables.
/sbin/iptables -I INPUT -p tcp -m tcp -j ACCEPT

echo "Rank: $NODE_RANK"

non_blocking_wait() {
  # https://www.baeldung.com/linux/background-process-get-exit-code
  local pid=$1
  local code=127 # special code to indicate not-finished
  if [[ ! -d "/proc/$pid" ]]; then
    wait "$pid"
    code=$?
  fi
  echo $code
}

wait_all_success_or_exit() {
  # https://www.baeldung.com/linux/background-process-get-exit-code
  local pids=("$@")
  while [[ ${#pids[@]} -ne 0 ]]; do
    all_success="true"
    for pid in "${pids[@]}"; do
      code=$(non_blocking_wait "$pid")
      if [[ $code -ne 127 ]]; then
        if [[ $code -ne 0 ]]; then
          echo "PID $pid failed with exit code $code"
          exit "$code"
        fi
      else
        all_success="false"
      fi
    done
    if [[ $all_success == "true" ]]; then
      echo "All pids succeeded"
      break
    fi
    sleep 5
  done
}


echo "Starting straggler detection on hosts ${MPI_HOSTS}"
echo "Sleeping for $BM_WAIT_TIME seconds before starting SD benchmark"

sleep "$BM_WAIT_TIME"
export RUN_ID="${N_NODES}node_${JOB_TIMESTAMP}"
export BM_LOG_DIR="sd_pipeline_log"

# Build Data Directories
ANALYSIS_DIR_SUFFIX='analysis'
DATA_DIR_SUFFIX='data'
OUTPUT_DIR="/gcs/${BM_LOG_DIR}/${RUN_ID}/ppbenchmark"
DATA_OUTPUT_DIR="${OUTPUT_DIR}/${DATA_DIR_SUFFIX}"
ANALYSIS_OUTPUT_DIR="${OUTPUT_DIR}/${ANALYSIS_DIR_SUFFIX}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${DATA_OUTPUT_DIR}"
mkdir -p "${ANALYSIS_OUTPUT_DIR}"
export DATA_OUTPUT_DIR

pids=()
export OMPI_COMM_WORLD_SIZE=$NPROCESS
for ((LOCAL_RANK=0; LOCAL_RANK < $ACCELERATORS_PER_POD; LOCAL_RANK++)); do
  export OMPI_COMM_WORLD_RANK=$(($ACCELERATORS_PER_POD*$NODE_RANK + $LOCAL_RANK))
  export OMPI_COMM_WORLD_LOCAL_RANK=$LOCAL_RANK
  echo "World rank:" ${OMPI_COMM_WORLD_RANK}
  /bin/bash /scripts/benchmark_wrapper.sh &
  last_pid=$!
  pids+=($last_pid)
  echo "Launched benchmark for rank $OMPI_COMM_WORLD_RANK on pid $last_pid"
done
wait_all_success_or_exit "${pids[@]}"

if [[ $NODE_RANK -eq 0 ]]; then
  echo "[${date}] Copying ppbenchmark data to ./tmp directory"
  TMP_DATA_DIR="./tmp/${DATA_DIR_SUFFIX}"
  TMP_ANALYSIS_DIR="./tmp/${ANALYSIS_DIR_SUFFIX}"
  mkdir -p "${TMP_DATA_DIR}"
  cp -r "${DATA_OUTPUT_DIR}" "./tmp"
  echo "[${date}] Starting ppbenchmark analysis"
  mkdir -p "${TMP_ANALYSIS_DIR}"
  python pp_benchmark_analysis_runner.py \
    --data_dir="${TMP_DATA_DIR}" \
    --output_dir="${TMP_ANALYSIS_DIR}" \
    --straggler_threshold_ms="${STRAGGLER_THRESHOLD_MS:-8}" \
    --interesting_event_offset="${interesting_event_offset:-4}"
  echo "$ls ./tmp/analysis"
  echo "[${date}] ppbenchmark analysis complete. Copying results to ${OUTPUT_DIR}"
  cp -r "./tmp/analysis" "${OUTPUT_DIR}"
  echo "Results at: https://pantheon.corp.google.com/storage/browser/${GCS_BUCKET}/${BM_LOG_DIR}/${RUN_ID}/ppbenchmark/"
fi
