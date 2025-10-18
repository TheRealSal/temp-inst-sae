#!/bin/bash

# Check if a seed is provided as an argument
if [ $# -ne 1 ]; then
  echo "Usage: $0 <seed>"
  exit 1
fi

SEED=$1
GPU=0

# Run for different dimensions
for DIM in 64 128 256 512 1024; do
  echo "Running with dimension $DIM and seed $SEED"
  CUDA_VISIBLE_DEVICES=$GPU python scratch.py --dim $DIM --seed $SEED
  sleep 1
done