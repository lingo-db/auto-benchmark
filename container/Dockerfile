FROM ubuntu:noble AS devimg
RUN apt-get update && apt-get -y install git python3 python3-pip
RUN pip install --break-system-packages numpy scipy
WORKDIR /app
COPY run-benchmark.py .
