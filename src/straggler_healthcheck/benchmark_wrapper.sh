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

#!/usr/bin/env bash
set -e
set -u
set -o pipefail

case $OMPI_COMM_WORLD_LOCAL_RANK in
        0) cstart=0; cend=13 ;;
        1) cstart=13; cend=26 ;;
        2) cstart=26; cend=39 ;;
        3) cstart=39; cend=52 ;;
        4) cstart=52; cend=65 ;;
        5) cstart=65; cend=78 ;;
        6) cstart=78; cend=91 ;;
        7) cstart=91; cend=104 ;;
        *) exit 1 ;;
esac

vstart=$((cstart+104))
vend=$((cend+104))

GPU_SERIAL=$(nvidia-smi --query-gpu serial --format csv,noheader -i "$OMPI_COMM_WORLD_LOCAL_RANK")
VM_ID=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/id?alt=text" -H "Metadata-Flavor: Google")
export GPU_SERIAL
export VM_ID
echo "VM_ID: ${VM_ID}; GPU_SERIAL: ${GPU_SERIAL}"
taskset --cpu-list $cstart-$((cend-1)),$vstart-$((vend-1)) \
        python /scripts/pp_benchmark_runner.py \
        --message_sizes_mb="${MESSAGE_SIZES_MB}" \
        --ompi_comm_world_rank="${OMPI_COMM_WORLD_RANK}" \
        --ompi_comm_world_local_rank="${OMPI_COMM_WORLD_LOCAL_RANK}" \
        --main_address="${CONTROLLER_ADDR}" \
        --hostname="${HOSTNAME}" \
        --n_nodes="${N_NODES}" \
        --n_batch="${N_BATCH}" \
        --n_microbatch="${N_MICROBATCH}" \
        --bidirectional="${BIDIRECTIONAL}" \
        --output_dir="${DATA_OUTPUT_DIR}"