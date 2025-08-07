FROM ubuntu:noble AS devimg
RUN apt-get update && apt-get -y install  git python3
WORKDIR /app
COPY run-benchmark.py .
