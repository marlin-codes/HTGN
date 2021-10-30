#!/usr/bin/env bash
device_id=0
for dataset in enron10 dblp
do
  for model in GRUGCN
  do
    python main.py \
          --model=${model} \
          --dataset=${dataset} \
          --device_id=${device_id}
  done
done