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

FROM nvidia/dcgm:3.3.5-1-ubuntu22.04
RUN apt-get update && apt-get install -y ca-certificates curl python3 python3-pip
WORKDIR /app
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && chmod +x kubectl
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

RUN pip install kubernetes

COPY src/gpu_healthcheck/dcgm.proto /app/
RUN pip install grpcio-tools
RUN pip install google-cloud-storage
RUN python3 -m grpc_tools.protoc -I /app/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /app/dcgm.proto

COPY src/gpu_healthcheck/gpu_healthcheck.py .
COPY src/checker_common.py .
ENV PYTHONUNBUFFERED=1

RUN chmod +x /app/gpu_healthcheck.py /app/checker_common.py
ENTRYPOINT ["python3", "/app/gpu_healthcheck.py"]