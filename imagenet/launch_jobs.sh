#!/bin/bash
IMAGENET_DIR=${IMAGENET_DIR:-/data/local/packages/ai-group.imagenet-full-size/prod/imagenet_full_size/}
NUM_LOCAL=$1
NUM_REPLICAS=${2:-$1}
LR=${3:-0.1}
RANK_START=${4:-0}
if [ -z $NUM_LOCAL ]; then
  echo "usage: ./launch_jobs.sh NUM_LOCAL [NUM_REPLICAS] [LR] [RANK_START]"
  exit 1
fi
for i in $(seq 0 $((NUM_LOCAL-1)))
do
  rank=$((i+RANK_START))
  device=$i
  echo "rank $rank device $device"
  CHECKPOINT_DIR=replica$i
  python3 -u main.py -a resnet18 --rank $rank --device $device --num-replicas $NUM_REPLICAS -b 32 --lr $LR --checkpoint-dir $CHECKPOINT_DIR $IMAGENET_DIR &>replica$i.log &
done
