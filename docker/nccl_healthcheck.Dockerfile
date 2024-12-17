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

FROM us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpx/nccl-plugin-gpudirecttcpx-nightly-cuda12.0:2024_03_04
WORKDIR /scripts

RUN apt-get update &&\
    apt-get install -y openssh-server ca-certificates curl net-tools

RUN mkdir /var/run/sshd && chmod 0755 /var/run/sshd && \
    ssh-keygen -t rsa -f /root/.ssh/google_compute_engine -b 2048 -P "" && \
    cp /root/.ssh/google_compute_engine.pub /root/.ssh/authorized_keys && \
    chmod 644 /root/.ssh/authorized_keys && \
    chmod 644 /root/.ssh/google_compute_engine.pub

RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" &&\
   chmod +x kubectl

RUN apt-get update && \
    apt-get install -y sudo software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.10 python3.10-distutils ca-certificates curl && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

COPY src/nccl_healthcheck/config.proto /scripts/
RUN pip install grpcio-tools
RUN pip install kubernetes
RUN pip install google-cloud-storage
RUN python3 -m grpc_tools.protoc -I /scripts/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /scripts/config.proto

COPY src/nccl_healthcheck/nccl_startup.py /scripts/
COPY src/nccl_healthcheck/config.py /scripts/
COPY src/nccl_healthcheck/run-nccl-combined-plugins.sh .
COPY src/checker_common.py /scripts/
RUN chmod +x /scripts/run-nccl-combined-plugins.sh
ENV PYTHONUNBUFFERED=1

RUN apt-get install -y dnsutils

ENTRYPOINT ["python3", "/scripts/nccl_startup.py"]