#!/bin/bash
IMAGENET_DIR=${IMAGENET_DIR:-/data/local/packages/ai-group.imagenet-full-size/prod/imagenet_full_size/}
NUM_LOCAL=$1
NUM_REPLICAS=${2:-$1}
LR=${3:-0.1}
if [ -z $NUM_LOCAL ]; then
  echo "usage: ./launch_jobs.sh NUM_LOCAL [NUM_REPLICAS] [LR]"
  exit 1
fi
RESUME=/mnt/vol/gfsai-east/ai-group/users/sgross/resnet-nccl2/resnet18-epoch5.pt
for i in $(seq 0 $((NUM_LOCAL-1)))
do
  echo $i
  CHECKPOINT_DIR=replica$i
  python3 -u main.py -a resnet18 --num-replicas $NUM_REPLICAS -b 64 --resume $RESUME --lr $LR --checkpoint-dir $CHECKPOINT_DIR $IMAGENET_DIR &>replica$i.log &
done
wait
