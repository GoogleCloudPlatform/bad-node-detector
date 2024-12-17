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

FROM nvcr.io/nvidia/pytorch:23.05-py3

# Set up GCSfuse
ENV GCSFUSE_VERSION=1.2.0
WORKDIR /scripts

RUN apt-get update && apt-get install --yes --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    iptables iproute2 ethtool sysstat \
  && echo "deb http://packages.cloud.google.com/apt gcsfuse-focal main" \
    | tee /etc/apt/sources.list.d/gcsfuse.list \
  && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
  && curl -LJO "https://github.com/GoogleCloudPlatform/gcsfuse/releases/download/v${GCSFUSE_VERSION}/gcsfuse_${GCSFUSE_VERSION}_amd64.deb" \
  && apt-get -y install fuse \
  && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  && dpkg -i "gcsfuse_${GCSFUSE_VERSION}_amd64.deb" \
  && mkdir /gcs


# Set up ssh
RUN apt-get update && apt install -y openssh-server &&\
  mkdir /var/run/sshd && chmod 0755 /var/run/sshd &&\
  ssh-keygen -t rsa -f /root/.ssh/google_compute_engine -b 2048 -P "" &&\
  cp /root/.ssh/google_compute_engine.pub /root/.ssh/authorized_keys &&\
  chmod 644 /root/.ssh/authorized_keys &&\
  chmod 644 /root/.ssh/google_compute_engine.pub

RUN echo -e \
"Host * \n\
  StrictHostKeyChecking no \n\
  User root \n\
  IdentityFile /root/.ssh/google_compute_engine \n\
  Port 2222" \
>> /root/.ssh/config &&\
sed -i "s/#Port 22/Port 2222/" /etc/ssh/sshd_config

COPY src/straggler_healthcheck/straggler_detection_healthcheck.proto .
COPY src/straggler_healthcheck/entrypoint.sh .
COPY src/straggler_healthcheck/benchmark_wrapper.sh .
COPY src/straggler_healthcheck/pp_benchmark_results_log.py .
COPY src/straggler_healthcheck/pp_benchmark.py .
COPY src/straggler_healthcheck/pp_benchmark_runner.py .
COPY src/straggler_healthcheck/pp_benchmark_analysis.py .
COPY src/straggler_healthcheck/pp_benchmark_analysis_runner.py .

RUN pip install protobuf
RUN pip install absl-py
RUN pip install grpcio-tools
RUN pip install google-cloud-storage
RUN python3 -m grpc_tools.protoc -I . --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions straggler_detection_healthcheck.proto

RUN chmod +x /scripts/entrypoint.sh /scripts/benchmark_wrapper.sh 
ENTRYPOINT ["/bin/bash", "/scripts/entrypoint.sh"]