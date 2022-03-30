#!/usr/bin/env bash

# train
#CUDA_VISIBLE_DEVICES=2 python train.py --model hdnet --backbone resnet50 --dataset inria --name wei-20-lr001  --lr 0.01 --aux-weight 20  --epochs 200 --batch-size 8  --save-epoch 5
#CUDA_VISIBLE_DEVICES=2 python train.py --model hdnet --backbone resnet50 --dataset whu_aerial --name wei-20-lr001  --lr 0.01 --aux-weight 20  --epochs 200 --batch-size 8  --save-epoch 5
#CUDA_VISIBLE_DEVICES=3 python train.py --model hdnet --backbone resnet50 --dataset whu_satellite --name wei-20-lr001  --lr 0.01 --aux-weight 20  --epochs 200 --batch-size 8  --save-epoch 5




