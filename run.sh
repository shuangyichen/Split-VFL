#!/bin/bash
set -x

WORKER_INDEX=$1
NUM_CLIENTS=2

python3 ./main.py \
  --party_idx $WORKER_INDEX \
  --client_num $NUM_CLIENTS \
  --grpc_ipconfig_path "/home/ubuntu/Split-VFL/grpc_ipconfig.csv"